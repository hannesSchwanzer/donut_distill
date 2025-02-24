import argparse
from datetime import datetime
from pathlib import Path
import torch
import wandb
from tqdm import tqdm
from transformers import GenerationConfig

from donut_distill.config.loader import load_config
from donut_distill.models.helpers import prepare_model_and_processor
from donut_distill.models.student import create_student_small
from donut_distill.training.utils import prepare_dataloader, prepare_optimizer_and_scheduler
from donut_distill.training.losses import calculate_loss_and_accuracy_distillation
from donut_distill.evaluation.evaluate import evaluate_docvqa
import donut_distill.config.config as CONFIG

TOKENIZERS_PARALLELISM = False

def train_one_epoch(model, student_model, train_dataloader, optimizer, scheduler, device, processor, start_step):
    """
    Train for one epoch with mid-epoch validation and logging.
    """
    # Set proper mode: if distilling, use teacher model in eval() and student in train()
    if CONFIG.DISTILL:
        model.eval()
        student_model.train()
    else:
        model.train()

    total_loss = 0.0
    scaler = torch.amp.GradScaler("cuda")
    steps = start_step
    num_batches = len(train_dataloader)
    val_check_interval = max(1, int(num_batches * CONFIG.VAL_CHECK_INTERVAL))

    for i, batch in enumerate(tqdm(train_dataloader, desc="Training Epoch")):
        # Unpack and send data to device
        pixel_values, decoder_input_ids, labels = [x.to(device) for x in batch]
        decoder_input_ids = decoder_input_ids[:, :-1]
        labels = labels[:, 1:]
        
        with torch.autocast(device_type="cuda"):
            if CONFIG.DISTILL:
                teacher_outputs = model(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    labels=labels,
                    output_attentions=True,
                    output_hidden_states=True,
                )
                student_outputs = student_model(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    labels=labels,
                    output_attentions=True,
                    output_hidden_states=True,
                )
                loss = calculate_loss_and_accuracy_distillation(
                    outputs=student_outputs,
                    teacher_outputs=teacher_outputs,
                    is_first_distillation_phase=True,
                    is_1phase_distillation=True,
                    decoder_layer_map=CONFIG.DECODER_LAYER_MAP,
                    device=device
                )
            else:
                outputs = model(pixel_values, decoder_input_ids=decoder_input_ids, labels=labels)
                loss = outputs.loss

        scaler.scale(loss).backward()

        if (i + 1) % CONFIG.ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.GRADIENT_CLIP_VAL)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item()
        steps += 1

        # Log training metrics at configured intervals
        if steps % CONFIG.LOG_INTERVAL == 0:
            wandb.log({
                "train/loss": loss.item(),
                "gpu/memory_allocated": torch.cuda.memory_allocated(),
                "gpu/memory_reserved": torch.cuda.memory_reserved(),
                "lr": optimizer.param_groups[0]["lr"],
                "step": steps,
            }, step=steps)

        # Mid-epoch validation
        if (i + 1) % val_check_interval == 0:
            # Switch to evaluation mode
            if CONFIG.DISTILL:
                student_model.eval()
            else:
                model.eval()
            torch.cuda.empty_cache()
            
            eval_results = evaluate_docvqa(
                model=student_model if CONFIG.DISTILL else model,
                processor=processor,
                device=device,
                val_dataloader=val_dataloader,
                generation_config=GenerationConfig(
                    early_stopping=True,
                    num_beams=1,
                ),
            )
            # Log evaluation metrics mid-epoch
            wandb.log(eval_results, step=steps)
            
            # Return to training mode
            if CONFIG.DISTILL:
                student_model.train()
            else:
                model.train()
            torch.cuda.empty_cache()

    avg_loss = total_loss / num_batches
    return avg_loss, steps

def validate(model, val_dataloader, processor, device):
    """
    Validate the model on the validation dataset.
    """
    model.eval()
    torch.cuda.empty_cache()

    with torch.autocast(device_type="cuda"):
        eval_results = evaluate_docvqa(
            model=model,
            processor=processor,
            device=device,
            val_dataloader=val_dataloader,
            generation_config=GenerationConfig(
                early_stopping=True,
                num_beams=1,
            ),
        )

    torch.cuda.empty_cache()
    return eval_results

def check_gradients(model):
    highest_gradient = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            if grad_norm > highest_gradient:
                highest_gradient = grad_norm
            if grad_norm > 1e2:  # Threshold for large gradients
                print(f"⚠️ High gradient detected in {name}: {grad_norm:.2f}")
    return highest_gradient

def train():
    model, processor, donut_config = prepare_model_and_processor(
        special_tokens=["<yes/>", "<no/>"], return_config=True, load_teacher=CONFIG.DISTILL
    )

    train_dataloader, val_dataloader = prepare_dataloader(model, processor)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if CONFIG.DISTILL:
        student_model = create_student_small(
            teacher=model,
            teacher_config=donut_config,
            encoder_layer_map=CONFIG.ENCODER_LAYER_MAP,
            decoder_layer_map=CONFIG.DECODER_LAYER_MAP,
        )
        student_model.to(device)

    model.to(device)

    # Optimizer and Scheduler
    optimizer, scheduler = prepare_optimizer_and_scheduler(
        model, len(train_dataloader.dataset)
    )

    # Logger
    wandb.init(
        project="donut-distill-docvqa",
        name="docvqa",
        config={
            "learning_rate": CONFIG.LR,
            "architecture": "Donut",
            "dataset": "docvqa",
            "epochs": CONFIG.MAX_EPOCHS,
            "gradient_clip_val": CONFIG.GRADIENT_CLIP_VAL,
        },
    )

    # Create directories for model and processor
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(CONFIG.RESULT_PATH) / f"donut_{timestamp}"

    scaler = torch.amp.GradScaler("cuda")
    best_val_metric = 0.0
    steps = 0
    num_batches_per_epoch = len(train_dataloader)
    val_check_interval_batches = max(
        1, int(num_batches_per_epoch * CONFIG.VAL_CHECK_INTERVAL)
    )

    for epoch in range(CONFIG.MAX_EPOCHS):
        # Training phase
        if CONFIG.DISTILL:
            model.eval()
            student_model.train()
        else:
            model.train()
        total_loss = 0
        for i, batch in enumerate(
            tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")
        ):
            pixel_values, decoder_input_ids, labels = batch
            pixel_values = pixel_values.to(device)
            decoder_input_ids = decoder_input_ids[:, :-1].to(device)
            labels = labels[:, 1:].to(device)

            with torch.autocast(device_type="cuda"):
                if CONFIG.DISTILL:
                    teacher_outputs = model(
                        pixel_values,
                        decoder_input_ids=decoder_input_ids,
                        labels=labels,
                        output_attentions=True,
                        output_hidden_states=True,
                    )
                    student_outputs = student_model(
                        pixel_values,
                        decoder_input_ids=decoder_input_ids,
                        labels=labels,
                        output_attentions=True,
                        output_hidden_states=True,
                    )
                    loss = calculate_loss_and_accuracy_distillation(
                        outputs=student_outputs, 
                        teacher_outputs=teacher_outputs,
                        is_first_distillation_phase=True,
                        is_1phase_distillation=True,
                        decoder_layer_map=CONFIG.DECODER_LAYER_MAP,  # Teacher has 4 Layers
                        device=device
                    )

                else:
                    outputs = model(
                        pixel_values, decoder_input_ids=decoder_input_ids, labels=labels
                    )
                    loss = outputs.loss

            scaler.scale(loss).backward()
            highest_gradient = check_gradients(model)

            if (i + 1) % CONFIG.ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), CONFIG.GRADIENT_CLIP_VAL
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            # Log training metrics
            if steps % CONFIG.LOG_INTERVAL == 0:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "gpu/memory_allocated": torch.cuda.memory_allocated(),
                        "gpu/memory_reserved": torch.cuda.memory_reserved(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        "highest_gradient": highest_gradient,
                    },
                    step=steps,
                )

            total_loss += loss.item()
            steps += 1

            if (i + 1) % val_check_interval_batches == 0:
                if CONFIG.DISTILL:
                    student_model.eval()
                else:
                    model.eval()
                torch.cuda.empty_cache()

                with torch.autocast(device_type="cuda"):
                    if CONFIG.DISTILL:
                        eval_results = evaluate_docvqa(
                            model=student_model,
                            processor=processor,
                            device=device,
                            val_dataloader=val_dataloader,
                            generation_config=GenerationConfig(
                                early_stopping=True,
                                num_beams=1,
                            ),
                        )
                    else:
                        eval_results = evaluate_docvqa(
                            model=model,
                            processor=processor,
                            device=device,
                            val_dataloader=val_dataloader,
                            generation_config=GenerationConfig(
                                early_stopping=True,
                                num_beams=1,
                            ),
                        )

                eval_results.update({"epoch": epoch})

                wandb.log(
                    eval_results,
                    step=steps,
                )

                if best_val_metric < eval_results["eval/anls"]:
                    print("Saving Model!")
                    best_val_metric = eval_results["eval/anls"]
                    model.save_pretrained(model_dir)
                    processor.save_pretrained(model_dir)

                torch.cuda.empty_cache()
                if CONFIG.DISTILL:
                    student_model.train()
                else:
                    model.train()

        avg_train_loss = total_loss / len(train_dataloader)

        log_data = {"train/avg_loss": avg_train_loss}
        log_data.update({"epoch": epoch})

        wandb.log(
            log_data,
            step=steps,
        )

        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Input the path to the config file with the settings you want to train with", type=str, default=None)
    args = parser.parse_args()

    if args.config:
        load_config(args.config)

    train()
