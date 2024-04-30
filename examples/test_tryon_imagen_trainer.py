import wandb
wandb.init(
    # set the wandb project where this run will be logged
    project="mynewproject_name"
)

from pathlib import Path
import torch
from torch.utils.data import DataLoader

# run something pip related to fix reqs
# run pip install mediapipe
from tryondiffusion import TryOnImagen, TryOnImagenTrainer, get_unet_by_name, SyntheticTryonDataset, tryondiffusion_collate_fn, SyntheticTryonDatasetFromDisk

TRAIN_UNET_NUMBER = 1
# BASE_UNET_IMAGE_SIZE = (64, 64)
# SR_UNET_IMAGE_SIZE = (256, 256)
BASE_UNET_IMAGE_SIZE = (128, 128)
SR_UNET_IMAGE_SIZE = (256, 256)
# BATCH_SIZE =2 
BATCH_SIZE =8
GRADIENT_ACCUMULATION_STEPS = 2
NUM_ITERATIONS = 500000
# NUM_ITERATIONS = 10
TIMESTEPS = (500, 500)

exp_name = Path('/home/roman/tryondiffusion_implementation/tryondiffusion_danny/experiments/overfit128_small200_batch_0')
exp_name.mkdir(parents=True, exist_ok=True)
samples_path = Path(exp_name, 'samples')
samples_path.mkdir(parents=True, exist_ok=True)

save_every_steps=500
sample_every = 1000

# save_every_steps=4
# sample_every = 4

def main():
    print("Instantiating the dataset and dataloader...")
    # dataset = SyntheticTryonDataset(
    #     num_samples=500, image_size=SR_UNET_IMAGE_SIZE if TRAIN_UNET_NUMBER == 2 else BASE_UNET_IMAGE_SIZE
    # )
    dataset = SyntheticTryonDatasetFromDisk(200)
    print(len(dataset))
    train_dataloader = DataLoader(
        dataset,
        num_workers=1,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=tryondiffusion_collate_fn,
    )
    validation_dataloader = DataLoader(
        dataset,
        num_workers=1,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=tryondiffusion_collate_fn,
    )
    print("Checking the dataset and dataloader...")
    sample = next(iter(train_dataloader))
    for k, v in sample.items():
        print(f"{k}: {v.shape}")

    # Instantiate the unets
    print("Instantiating U-Nets...")
    base_unet = get_unet_by_name("base")
    sr_unet = get_unet_by_name("sr")
    
    base_unet.to('cuda')

    # Instantiate the Imagen model
    imagen = TryOnImagen(
        unets=(base_unet, sr_unet),
        image_sizes=(BASE_UNET_IMAGE_SIZE, SR_UNET_IMAGE_SIZE),
        timesteps=TIMESTEPS,
    )

    print("Instantiating the trainer...")
    trainer = TryOnImagenTrainer(
        # init_checkpoint_path='/home/roman/tryondiffusion_implementation/tryondiffusion_danny/experiments/overfit128_small200_batch_0/checkpoint.4000.pt',
        checkpoint_path=str(exp_name),
        checkpoint_every=save_every_steps,
        imagen=imagen,
        max_grad_norm=1.0,
        # accelerate_cpu=True,
        accelerate_cpu=False,
        accelerate_gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    )

    trainer.add_train_dataloader(train_dataloader)
    trainer.add_valid_dataloader(validation_dataloader)

    print("Starting training loop...")
    # training loop
    from tqdm import trange
    for i in trange(NUM_ITERATIONS):
        # TRAINING
        loss = trainer.train_step(unet_number=TRAIN_UNET_NUMBER)
        # print(f"iter: {i}\nloss: {loss}")
        wandb.log({"train/loss": loss})

        if i % 10 == 0:
            valid_loss = trainer.valid_step(unet_number=TRAIN_UNET_NUMBER)
            # print(f"valid loss: {valid_loss}")
            wandb.log({"valid/loss": valid_loss})
            
        # if i % save_every_steps == 0:
            
        if i % sample_every == 0 and i!=0 or i+1==NUM_ITERATIONS:
            validation_sample = next(trainer.valid_dl_iter)
            _ = validation_sample.pop("person_images")
            imagen_sample_kwargs = dict(
                **validation_sample,
                batch_size=BATCH_SIZE,
                cond_scale=2.0,
                start_at_unet_number=1,
                return_all_unet_outputs=True,
                return_pil_images=True,
                use_tqdm=True,
                use_one_unet_in_gpu=True,
                stop_at_unet_number=1
            )
            images = trainer.sample(**imagen_sample_kwargs)  # returns List[Image]
            
            iter_samples_path = (samples_path / str(i))
            iter_samples_path.mkdir(parents=True, exist_ok=True)
            for idx_unet, unet_output in enumerate(images):
                for idx_step, image in enumerate(unet_output):
                    image.save(str(iter_samples_path / f'{idx_unet}_{idx_step}_sample.png'))
    wandb.finish()


if __name__ == "__main__":
    # python ./examples/test_tryon_imagen_trainer.py
    main()
    
    # PYTHONPATH=. python3 examples/test_tryon_imagen_trainer.py