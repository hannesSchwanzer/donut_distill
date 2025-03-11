import argparse
from datetime import datetime
from pathlib import Path
import torch
import wandb
from tqdm import tqdm
from transformers import GenerationConfig

from donut_distill.config.loader import load_config
from donut_distill.models.helpers import prepare_model_and_processor
from donut_distill.models.student import create_student_small_with_encoder
from donut_distill.training.utils import prepare_dataloader, prepare_optimizer_and_scheduler
from donut_distill.training.losses import calculate_loss_and_accuracy_distillation
from donut_distill.evaluation.evaluate import evaluate_docvqa
import donut_distill.config.config as CONFIG

TOKENIZERS_PARALLELISM = False

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
                print(f"High gradient detected in {name}: {grad_norm:.2f}")
    return highest_gradient

def train():
    model, processor, donut_config = prepare_model_and_processor(
        special_tokens=["<yes/>", "<no/>"], return_config=True, load_teacher=CONFIG.DISTILL
    )

    train_dataloader, val_dataloader = prepare_dataloader(model, processor)

    # Update the config vocab size, not sure if needed
    donut_config.vocab_size = len(processor.tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if CONFIG.DISTILL:
        student_model = create_student_small_with_encoder(
            teacher=model,
            teacher_config=donut_config,
            encoder_layer_map=CONFIG.ENCODER_LAYER_MAP,
            decoder_layer_map=CONFIG.DECODER_LAYER_MAP,
        )
        student_model.to(device)

    model.to(device)

    # Optimizer and Scheduler
    optimizer, scheduler = prepare_optimizer_and_scheduler(
        model=student_model if CONFIG.DISTILL else model,
        len_trainingsdata=len(train_dataloader.dataset)
    )

    # Logger
    log_config = {
            "learning_rate": CONFIG.LR,
            "architecture": "Donut",
            "dataset": "docvqa",
            "epochs": CONFIG.MAX_EPOCHS,
            "gradient_clip_val": CONFIG.GRADIENT_CLIP_VAL,
        }
    if CONFIG.DISTILL:
        log_config.update({
            "teacher_path": CONFIG.TEACHER_MODEL_PATH,
            "alpha": CONFIG.ALPHA,
            "beta": CONFIG.BETA,
            "gamma": CONFIG.GAMMA,
            "delta": CONFIG.DELTA,
            "encoder_weight": CONFIG.ENCODER_WEIGHT,
            "decoder_weight": CONFIG.DECODER_WEIGHT,
            "encoder_layer_map": CONFIG.ENCODER_LAYER_MAP,
            "decoder_layer_map": CONFIG.DECODER_LAYER_MAP,
        })

    wandb.init(
        project="donut-distill-docvqa",
        name=CONFIG.WANDB_NAME,
        config=log_config,
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

    # Validate before training
    if CONFIG.DISTILL:
        student_model.eval()
    else:
        model.eval()

    with torch.autocast(device_type="cuda"):
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
    wandb.log(
        eval_results,
        step=steps,
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
                    losses = calculate_loss_and_accuracy_distillation(
                        outputs=student_outputs, 
                        teacher_outputs=teacher_outputs,
                        is_first_distillation_phase=True,
                        is_1phase_distillation=True,
                        decoder_layer_map=CONFIG.DECODER_LAYER_MAP,  # Teacher has 4 Layers
                        device=device,
                        alpha=CONFIG.ALPHA,
                        beta=CONFIG.BETA,
                        gamma=CONFIG.GAMMA,
                        delta=CONFIG.DELTA,
                        encoder_weight=CONFIG.ENCODER_WEIGHT,
                        decoder_weight=CONFIG.DECODER_WEIGHT,
                    )
                    loss = losses['total_loss']

                else:
                    outputs = model(
                        pixel_values, decoder_input_ids=decoder_input_ids, labels=labels
                    )
                    loss = outputs.loss

            scaler.scale(loss).backward()
            # highest_gradient = check_gradients(model=student_model if CONFIG.DISTILL else model)

            if (i + 1) % CONFIG.ACCUMULATION_STEPS == 0:
                # Unscale gradients before clipping
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    student_model.parameters() if CONFIG.DISTILL else model.parameters(), CONFIG.GRADIENT_CLIP_VAL
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                steps += 1

            # Log training metrics
            if i % CONFIG.LOG_INTERVAL == 0:
                log_data = {
                        "train/loss": loss.item(),
                        "gpu/memory_allocated": torch.cuda.memory_allocated(),
                        "gpu/memory_reserved": torch.cuda.memory_reserved(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        # "highest_gradient": highest_gradient,
                    }
                if CONFIG.DISTILL:
                    log_data.update(
                        # Filter out total loss (gets added anyway and is a tensor)
                        {k: v for k, v in losses.items() if k != "total_loss"}
                    )
                wandb.log(
                    log_data,
                    step=steps,
                )

            total_loss += loss.item()

            if (i + 1) % val_check_interval_batches == 0:
                if CONFIG.DISTILL:
                    student_model.eval()
                else:
                    model.eval()
                torch.cuda.empty_cache()

                with torch.autocast(device_type="cuda"):
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

                eval_results.update({"epoch": epoch})

                wandb.log(
                    eval_results,
                    step=steps,
                )

                if best_val_metric < eval_results["eval/anls"]:
                    print("Saving Model!")
                    best_val_metric = eval_results["eval/anls"]
                    if CONFIG.DISTILL:
                        student_model.save_pretrained(model_dir)
                    else:
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
