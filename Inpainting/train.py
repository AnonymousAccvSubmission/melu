import argparse
from datetime import datetime
import os
import yaml
from torch import nn
import logging 
import torch
from PIL import Image
import numpy as np

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

from data import define_dataloader
from models import get_generator, Discriminator256, Inception
from torchsummary import summary
from utils import denormalize, set_seed
from eval import calcu_eval
from models.loss import PerceptualLoss, GANLoss
from models import sed
import wandb


t_start = datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, default="./configs/celeba_inpaintingV3.yaml", help="the config files")
parser.add_argument("--resume", type=int, default=0, help="epoch to start training from")
parser.add_argument("--epochs", type=int, default=150, help="number of epochs of training")
parser.add_argument("--hr_shape", type=int, default=256, help="training image size 256 or 512")
parser.add_argument("--lambda_adv", type=float, default=0.01, help="adversarial loss weight")
parser.add_argument("--lambda_l1", type=float, default=2, help="l1 loss weight")
parser.add_argument("--out_dir", default='', help="where to store the output")
parser.add_argument("--gpu", type=int, default=0, help="gpu number")
opt = parser.parse_args()


config = yaml.load(open(opt.cfg, encoding="utf-8"), Loader=yaml.Loader)
if opt.out_dir != "":
    config["out_dir"] = opt.out_dir

# set global random seeds
set_seed(config["seed"])

# set sample and models save dir
img_dir = os.path.join(config["out_dir"], "images")
model_dir = os.path.join(config["out_dir"], "models")

os.makedirs(img_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

logging.basicConfig(filename=os.path.join(config["out_dir"], "train.log"), level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

if torch.cuda.is_available():
    torch.cuda.set_device(1)
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')

img_size = opt.hr_shape
# Initialize generator and discriminator
generator = get_generator(config["models"]["generator"]).to(device)
#discriminator = Discriminator256(4).to(device)
# fname = "./metrics/inception-2015-12-05.pt"
# incep = torch.jit.load(open(fname, "rb")).cpu()
incep = Inception().cpu()
incep.eval()

# Summary of the networks
summary(generator, [(3, img_size, img_size), (1, img_size, img_size), (1, img_size, img_size)])
#summary(discriminator, [(3, img_size, img_size), (1, img_size, img_size)])
# summary(ie, (3, hr_shape, hr_shape))



# Set feature extractor to inference mode
#incep.eval()

# Losses
vgg_loss = PerceptualLoss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)
loss_gan = GANLoss(gan_type='vanilla').to(device)

wandb.init(project = 'AGGNet_training',
               config = {
                   "leanring_rate":config["optim"]["lr_g"],
                    "batch_size" : config["train"]["batch_size"],
                   "train_val_dataset" : "ffhq_65k",
                    "gan_loss" : "Perceptual+L1+Adv",
                   "model_type" : config["models"]["generator"]["name"],
               }
               )


model_ex_config = config['model_ex']
model_ex = sed.CLIP_Semantic_extractor(model_ex_config['layers'], model_ex_config['pretrained'], model_ex_config['path'], model_ex_config['output_dim'], model_ex_config['heads']).to('cuda')
discriminator = sed.SeD_P(**config['model_d']).to(device)

if config["resume"] != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load(model_dir+"/generator_%d.pth" % config["resume"]))
    discriminator.load_state_dict(torch.load(model_dir+"/discriminator_%d.pth" % config["resume"]))
    print("models loaded")


#model_ex = sed.CLIP_Semantic_extractor(**model_ex_config).to('cuda')
#discriminator = sed.SeD_P(**model_d_config).to('cuda')
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=config["optim"]["lr_g"], betas=(config["optim"]["b1"], config["optim"]["b2"]))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config["optim"]["lr_d"], betas=(config["optim"]["b1"], config["optim"]["b2"]))

# load datasets
train_data, val_data = define_dataloader(config)

# ----------
#  Training
# ----------

# init best psnr metrice
best_psnr = 0
index = 0
pre_epoch = 2
for epoch in range(config["resume"], config["epoch"]):
    D_loss = 0
    G_loss = 0
    percep = 0
    adv = 0
    pixel = 0
    adv_loss_g  = 0
    adv_D_real = 0
    adv_D_fake = 0
    if epoch < pre_epoch:
        t = datetime.now()
        for i, imgs in enumerate(train_data):
            # Configure model input
            index += 1

            img = imgs["image"].to(device)
            prior = imgs["prior"].to(device)
            mask = imgs["mask"].to(device)
            img_mask = img * (1 - mask) + torch.rand(img.shape).to(device) * mask
            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # Generate a extended image from input
            gen_img = generator(img, prior, mask)

            loss_pixel = criterion_pixel(gen_img, img)

            loss_percep = vgg_loss(img, gen_img)
            # Total generator loss
            # loss_G = opt.lambda_adv * loss_GAN + loss_pixel * opt.lambda_l1 + loss_percep * 0.4

            loss_G =  loss_pixel * opt.lambda_l1 + loss_percep * 0.4
            loss_G.backward()
            optimizer_G.step()



            G_loss += loss_G.item()

            pixel += loss_pixel.item()
            percep += loss_percep.item()
            if i % 100 == 0:
                print(f"Epoch:{epoch + 1}/{opt.epochs}, iter: {i}, t/iter: {(datetime.now() - t_start) / (i + 1)}")
            if index % 100 == 0:
                wandb.log({"epoch": epoch, "gan_loss": loss_G.item(),
                           "perceptual_loss": loss_percep.item(), "Pixel_loss": loss_pixel.item()})
        avg_pixel_loss = pixel / len(train_data)
        avg_percep_loss = percep / len(train_data)



        if epoch % config["eval_inter"] == 0:
            generator.eval()
            with torch.no_grad():
                metrice = calcu_eval(generator, incep, val_data, device, is_ssim=True)
            logging.info(f"val dataset contain {len(val_data)} images")
            logging.info(
                "epoch: {}, fid score: {}, psnr score: {}, ssim: {}".format(epoch + 1, metrice["fid"], metrice["psnr"],
                                                                            metrice["ssim"]))
            wandb.log({"epoch": epoch, "PSNR": metrice["psnr"],
                       "SSIM": metrice["ssim"], "FID": metrice["fid"]})

            # save the best model
            if metrice["psnr"] > best_psnr:
                best_psnr = metrice["psnr"]
                torch.save(generator.state_dict(), model_dir + "/generator_best.pth")
            generator.train()

        if (epoch + 1) % config["sample_inter"] == 0:
            # Save example results
            img_grid = denormalize(torch.cat((img_mask, gen_img, img), -1))
            save_image(img_grid, img_dir + "/epoch-{}.png".format(epoch + 1), nrow=1, normalize=False)

        if (epoch + 1) % config["save_model"] == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), model_dir + "/generator_{}.pth".format(epoch + 1))
            logging.info(f"epoch {epoch + 1} model saved!")

        logging.info(
            'Epoch:{1}/{2} pixel:{3} percep_loss:{4} time:{0}'.format(
                datetime.now() - t, epoch + 1, opt.epochs, avg_pixel_loss, avg_percep_loss))

    else:
        t = datetime.now()
        for i, imgs in enumerate(train_data):
            # Configure model input
            index +=1

            img = imgs["image"].to(device)
            prior = imgs["prior"].to(device)
            mask = imgs["mask"].to(device)
            img_mask = img * (1-mask) + torch.rand(img.shape).to(device) * mask
            # ------------------
            #  Train Generators
            # ------------------
            for p in discriminator.parameters():
                p.requires_grad = False

            optimizer_G.zero_grad()

            # Generate a extended image from input
            gen_img = generator(img, prior, mask)

            # Measure pixel-wise loss against ground truth
            loss_pixel = criterion_pixel(gen_img, img)

            #pred_fake = discriminator(gen_img, mask)

            # Adversarial loss (relativistic average GAN)
            #loss_GAN = -pred_fake.mean()

            # if you are training with edges it should be given here along with image. concatenate and give
            #########################################################################
            sementic_model_in = (img + 1) / 2
            sementic_detail = model_ex(sementic_model_in)



            disc_mode_in = torch.cat([(gen_img + 1) / 2, mask], dim=1)
            real_d_pred = discriminator(disc_mode_in, sementic_detail)
            gen_loss_disc_real =loss_gan(real_d_pred, True, is_disc=False)
            ##########################################################################
            loss_percep = vgg_loss(img, gen_img)
            # Total generator loss
            #loss_G = opt.lambda_adv * loss_GAN + loss_pixel * opt.lambda_l1 + loss_percep * 0.4

            loss_G = opt.lambda_adv * gen_loss_disc_real + loss_pixel * opt.lambda_l1 + loss_percep * 0.4
            loss_G.backward()
            optimizer_G.step()

            for p in discriminator.parameters():
                p.requires_grad = True


            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            #pred_real = discriminator(img, mask)
            ##########################################################################################################
            disc_mode_in = torch.cat([(img + 1) / 2, mask], dim=1)
            real_d_pred = discriminator(disc_mode_in, sementic_detail)
            loss_disc_real = loss_gan(real_d_pred, True, is_disc=True)
            loss_disc_real.backward()

            disc_mode_in = torch.cat([(gen_img.detach().clone() + 1) / 2, mask], dim=1)
            fake_d_pred = discriminator(disc_mode_in, sementic_detail)
            loss_d_fake = loss_gan(fake_d_pred, False, is_disc=True)
            loss_d_fake.backward()

            ###########################################################################################################
            #pred_fake = discriminator(gen_img.detach(), mask)

            # Total loss
            #loss_D = nn.ReLU()(1.0 - pred_real).mean() + nn.ReLU()(1.0 + pred_fake).mean()

            #loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------
            #D_loss += loss_D.item()
            G_loss += loss_G.item()
            #adv += loss_GAN.item()
            pixel += loss_pixel.item()
            percep += loss_percep.item()
            adv_loss_g += gen_loss_disc_real.item()
            adv_D_real = loss_disc_real.item()
            adv_D_fake = loss_d_fake.item()
            if i % 100 == 0:
                print(f"Epoch:{epoch+1}/{opt.epochs}, iter: {i}, t/iter: {(datetime.now() - t_start) / (i+1)}")
            if index % 100 == 0:
                wandb.log({"epoch": epoch,  "gan_loss": loss_G.item(), "generator_loss_adv": gen_loss_disc_real,
                           "dis_loss_real": loss_disc_real, "dis_loss_fake": loss_d_fake, "perceptual_loss": loss_percep.item(), "Pixel_loss": loss_pixel.item()})
        #avg_D_loss = D_loss / len(train_data)
        #avg_G_loss = G_loss / len(train_data)
        avg_Generator_loss = adv_loss_g / len(train_data)
        avg_Discriminator_loss = adv_D_real / len(train_data)
        avg_adv_loss = adv / len(train_data)
        avg_pixel_loss = pixel / len(train_data)
        avg_percep_loss = percep / len(train_data)

        logging.info(
            'Epoch:{1}/{2} D_loss:{3} G_loss:{4} adv:{5} pixel:{6} percep_loss:{7} time:{0}'.format(
                datetime.now() - t, epoch + 1, opt.epochs, avg_Discriminator_loss,
                avg_Generator_loss, avg_adv_loss, avg_pixel_loss, avg_percep_loss))

        if epoch % config["eval_inter"] == 0:
            generator.eval()
            with torch.no_grad():
                metrice = calcu_eval(generator, incep, val_data, device, is_ssim=True)
            logging.info(f"val dataset contain {len(val_data)} images")
            logging.info("epoch: {}, fid score: {}, psnr score: {}, ssim: {}".format(epoch+1, metrice["fid"], metrice["psnr"], metrice["ssim"]))
            wandb.log({"epoch": epoch, "PSNR": metrice["psnr"],
                       "SSIM": metrice["ssim"], "FID": metrice["fid"]})


            # save the best model
            if metrice["psnr"] > best_psnr:
                best_psnr = metrice["psnr"]
                torch.save(generator.state_dict(), model_dir + "/generator_best.pth")
            generator.train()

        if (epoch + 1) % config["sample_inter"] == 0:
            # Save example results
            img_grid = denormalize(torch.cat((img_mask, gen_img, img), -1))
            save_image(img_grid, img_dir + "/epoch-{}.png".format(epoch + 1), nrow=1, normalize=False)

        if (epoch + 1) % config["save_model"] == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), model_dir + "/generator_{}.pth".format(epoch + 1))
            torch.save(discriminator.state_dict(), model_dir + "/discriminator_{}.pth".format(epoch + 1))
            logging.info(f"epoch {epoch+1} model saved!")

