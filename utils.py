import cv2, torch, os, skimage, torchvision, random

import json

import numpy                       as np
import albumentations              as A
import segmentation_models_pytorch as smp
import torch.nn                    as nn
import matplotlib.pyplot           as plt

from torch.utils.data       import Dataset
from albumentations.pytorch import ToTensorV2
from copy                   import deepcopy
from torch.utils.data       import DataLoader
from torchvision            import transforms
from tqdm                   import tqdm

IMGSZ = 512

class MyDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform   = transform
        self.images      = [x for x in os.listdir(os.path.join(dataset_dir, 'images')) if x.endswith(".png")]
                                 
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        
        try:
            image = cv2.imread(os.path.join(self.dataset_dir, 'images', self.images[index]), cv2.IMREAD_GRAYSCALE)
            # image = skimage.io.imread(os.path.join(self.dataset_dir, 'images', self.images[index]))
            
            mask  = cv2.imread(os.path.join(self.dataset_dir, 'masks', self.images[index].replace('.png','_mask.png')), cv2.IMREAD_GRAYSCALE)

            if self.transform is not None:
                transformed = self.transform(image=image, mask=mask)
                image, mask = transformed['image'], transformed['mask']
            
            return image, mask, self.images[index]
        
        except:
            image = cv2.imread(os.path.join(self.dataset_dir, 'images', self.images[index]), cv2.IMREAD_GRAYSCALE)
            # image = skimage.io.imread(os.path.join(self.dataset_dir, 'images', self.images[index]))

            if self.transform is not None:
                image = self.transform(image=image)['image']

            return image, self.images[index]

def get_train_transforms():
    return A.Compose(
                [
                    A.Resize(IMGSZ, IMGSZ, interpolation=cv2.INTER_NEAREST, always_apply=False, p=1),
                    A.HorizontalFlip(p=0.5),
                    # A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.1, always_apply=False, p=0.5),
                    A.VerticalFlip(p=0.25),
                    # A.ToFloat(),
                    
                    # A.Normalize(
                    #     mean = [0.485, 0.456, 0.406],
                    #     std  = [0.229, 0.224, 0.225],
                    #     max_pixel_value=255.0,
                    # ),
                    ToTensorV2(),
                ],
            )
            
def get_val_transforms():
    return A.Compose(
                [
                    A.Resize(IMGSZ, IMGSZ, interpolation=cv2.INTER_NEAREST, always_apply=False, p=1),
                    # A.ToFloat(),
                    # A.Normalize(
                    #     mean = [0.485, 0.456, 0.406],
                    #     std  = [0.229, 0.224, 0.225],
                    #     max_pixel_value=255.0,
                    # ),
                    ToTensorV2(),
                ],
            )


def build_dataloders(
    train_dir,
    val_dir,
    test_dir    = None,
    train_bs    = 5,
    val_bs      = 1,
    num_workers = 0,
    pin_memory  = True,
):
    train_ds = MyDataset(
                    dataset_dir = train_dir,
                    transform   = get_train_transforms(),
               )
    
    train_loader = DataLoader(
                       train_ds,
                       batch_size  = train_bs,
                       num_workers = num_workers,
                       pin_memory  = False,
                       shuffle     = True,
                   )

    val_ds = MyDataset(
                    dataset_dir = val_dir,
                    transform   = get_val_transforms(),
             )

    val_loader = DataLoader(
                     val_ds,
                     batch_size  = val_bs,
                     num_workers = num_workers,
                     pin_memory  = False,
                     shuffle     = False,
                 )
    
    if test_dir:
        test_ds = MyDataset(
                        dataset_dir = test_dir,
                        transform   = get_val_transforms(),
                 )

        test_loader = DataLoader(
                         test_ds,
                         batch_size  = val_bs,
                         num_workers = num_workers,
                         pin_memory  = False,
                         shuffle     = False,
                     )
    else:
        test_loader = None

    return train_loader, val_loader, test_loader

def get_random_color(classes):
    num = len(classes)
    gray_level = [0,255] if num <= 3 else [0,125,255] if num <= 15 else [0,50,100,150,200,250]
    colors = {}
    color_list = []
    while len(color_list) < num:
        color = random.choices(gray_level, k = 3)
        if color not in color_list:
            color_list.append(color)
            colors[classes[len(colors)]] = color

    return colors
def get_class_colors(classes):
    colors = {
        'FP': [128, 0, 128],    # Purple for class 1
        "FPD": [0, 255, 255],    # Cyan for class 2
        "ASH": [0, 128, 0],      # Green for class 3
        "bf": [255, 0, 0],      # Red for class 4
    }
    return colors


class Trainer:
    def __init__(
        self, 
        dataset_dir, 
        out_dir, 
        classes, 
        patience,
        encoder    = 'resnet34', 
        decoder    = 'Unet',
        optimizer  = 'Nadam',
        init_lr    = 0.0025,
        num_epochs = 100, 
        train_bs   = 5, 
        val_bs     = 1, 
        device     ='cuda'
    ):
        self.model = self._get_model(encoder=encoder, decoder=decoder, num_classes=len(classes)+1, device=device)
        self.opt   = self._get_optimizer(model=self.model, name=optimizer, init_lr=init_lr)
        
        self.train_loader, self.val_loader, self.test_loader = build_dataloders(
                                                                  train_dir = os.path.join(dataset_dir, 'train'),
                                                                  val_dir   = os.path.join(dataset_dir, 'train'),
                                                                  test_dir  = os.path.join(dataset_dir, 'test'),
                                                                  train_bs  = train_bs,
                                                                  val_bs    = val_bs,
                                                               )
        
        # self.loss        = nn.BCEWithLogitsLoss()
        self.loss        = nn.CrossEntropyLoss()
        self.scaler      = torch.cuda.amp.GradScaler()
        self.device      = device
        self.patience    = patience 
        self.num_epochs  = num_epochs
        self.classes     = classes
        self.dataset_dir = dataset_dir
        self.colors      = get_random_color(classes)
        
        os.makedirs(os.path.join(out_dir, 'dump'), exist_ok=True)
        runs = [x for x in os.listdir(os.path.join(out_dir, 'dump')) if '.' not in x]
        if len(runs) == 0:
            self.out_dir    = os.path.join(out_dir, 'dump/run_0')
            os.makedirs(os.path.join(out_dir, 'dump/run_0/evaluation'), exist_ok=True)
        else:
            idx = 0
            for name in runs:
                idx = max(idx, int(name.split('_')[-1]))
            self.out_dir    = os.path.join(out_dir, f'dump/run_{idx+1}')
            os.makedirs(os.path.join(out_dir, f'dump/run_{idx+1}/evaluation'), exist_ok=True)
        
    @staticmethod
    def _get_model(encoder, decoder, num_classes, device, encoder_weights="imagenet"):
        if decoder == 'Unet':
            model = smp.Unet(
                        encoder_name    = encoder,     
                        encoder_weights = encoder_weights,  
                        in_channels     = 1,               
                        classes         = num_classes,                   
                    )
            
        elif decoder == 'UnetPlusPlus':
            model = smp.UnetPlusPlus(
                        encoder_name    = encoder,     
                        encoder_weights = encoder_weights,  
                        in_channels     = 1,               
                        classes         = num_classes,                   
                    )
        
        model.to(device)
        
        return model
    
    @staticmethod
    def _get_optimizer(model, name, init_lr):
        if name == 'Adam':
            opt = torch.optim.Adam(
                        model.parameters(), 
                        lr           = init_lr, 
                        betas        = (0.9, 0.999), 
                        eps          = 1e-08, 
                        weight_decay = 0, 
                        amsgrad      = False,
                   )
        elif name == 'Nadam':
            opt = torch.optim.NAdam(
                        model.parameters(), 
                        lr            = init_lr, 
                        betas         =(0.9, 0.999), 
                        eps           = 1e-08, 
                        weight_decay  = 0, 
                        momentum_decay= 0.001, 
                        foreach       = None
                    )
            
        return opt
    
    def train_fn(self, epoch):
        loop = tqdm(self.train_loader)
        total_loss = 0
        for batch_idx, (data, targets, _) in enumerate(loop):
            data = data.to(device=self.device).float()
            targets = targets.long().to(device=self.device)

            # forward
            predictions = self.model(data)
            loss = self.loss(predictions, targets)

            # backward
            total_loss += loss.item()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # update tqdm loop
            loop.set_postfix(loss=total_loss/(batch_idx+1))
            
        return total_loss/len(self.train_loader)
          
    def _save_checkpoint(self, ckpt):
        print(f"=> Saving checkpoint with training loss = {ckpt['train_loss']}, dice_val = {ckpt['DICE_score_val']} and dice_test = {ckpt['DICE_score_test']}")
        torch.save(ckpt, os.path.join(self.out_dir, 'training_checkpoint.pth'))  
        
    def _save_single_pred(self, preds, name, is_test):
        rgb = cv2.imread(os.path.join(self.dataset_dir, 'test/images' if is_test else 'val/images', name[0]), cv2.IMREAD_COLOR)
        
        rgb = A.Compose(
                [
                    A.Resize(IMGSZ, IMGSZ, interpolation=cv2.INTER_NEAREST, always_apply=False, p=1),
                ],
            )(image=rgb)['image']
        
        for ch in range(3):
            for idx, cls_name in enumerate(self.classes):
                rgb[:,:,ch][preds[idx+1]!=0] = self.colors[cls_name][ch]
            
        skimage.io.imsave(os.path.join(self.out_dir, 'evaluation', f"pred_{name[0].split('.')[0]}.png"), rgb)
        
    def _save_predictions_as_imgs(self, loader, is_test=True, conf = 0.95):
        self.model.eval()
        if is_test:
            for idx, (x, _, name) in enumerate(loader):
                x = x.to(device=self.device).float()
                with torch.no_grad():
                    preds = torch.softmax(self.model(x).squeeze(0), dim=0)
                    preds = (preds > conf).float()
                    
                self._save_single_pred(preds.cpu().numpy(), name, is_test)

                    
        else:
            for idx, (x, y, name) in enumerate(loader):
                x = x.to(device=self.device).float()
                with torch.no_grad():
                    preds = torch.sigmoid(self.model(x))
                    preds = (preds > conf).float()
                    
                self._save_single_pred(preds.cpu().numpy(), name, is_test)

        
        self.model.train()
    
    def _trans_mask(self, y):
        mask = torch.zeros(len(self.classes)+1, IMGSZ, IMGSZ)
        for item in range(len(self.classes)+1):
            mask[item,:,:] = y == item
        
        return mask
    
    def _check_accuracy(self):
        num_correct = 0
        num_pixels = 0
        dice_score = 0
        self.model.eval()

        with torch.inference_mode():
          dice_score_c_loop = {'FP':0,'FPD':0,'ASH':0,'bf':0}
          num_correct_c = 0
          num_pixels_c = 0
          for x, y, _ in self.val_loader:
              x = x.to(self.device).float()
              y = self._trans_mask(y).to(self.device)
              preds = torch.softmax(self.model(x).squeeze(0), dim=0)
              preds = (preds > 0.5).float()

              num_correct += (preds[1:,:,:] == y[1:,:,:]).sum()
              num_pixels += torch.numel(preds[1:,:,:])
              dice_score += (2 * (preds[1:,:,:] * y[1:,:,:]).sum()) / ((preds[1:,:,:] + y[1:,:,:]).sum() + 1e-8)

              # Dice Score for each class               
              for c in range(1,len(y)):                  
                num_correct_c += (preds[c,:,:] == y[c,:,:]).sum()
                num_pixels_c += torch.numel(preds[c,:,:])
                # print((2 * (preds[c,:,:] * y[c,:,:]).sum()) / ((preds[c,:,:] + y[c,:,:]).sum() + 1e-8))
                dice_score_c_loop[self.classes[c-1]] += (2 * (preds[c,:,:] * y[c,:,:]).sum()) / ((preds[c,:,:] + y[c,:,:]).sum() + 1e-8)

        dice_score_c = {key: float(value / len(self.val_loader)) for key, value in dice_score_c_loop.items()}
        print(
            f"Prefomance: accuracy = {num_correct/num_pixels*100:.2f}, DICE score = {dice_score/len(self.val_loader)}, DICE_score_c = {dice_score_c}"
        )
        self.model.train()
        
        return dice_score/len(self.val_loader),dice_score_c

    def _check_accuracy_test(self):
        num_correct = 0
        num_pixels = 0
        dice_score = 0
        self.model.eval()

        with torch.inference_mode():
            dice_score_c_loop = {'FP':0,'FPD':0,'ASH':0,'bf':0}
            num_correct_c = 0
            num_pixels_c = 0
            for x, y, _ in self.test_loader:
                x = x.to(self.device).float()
                y = self._trans_mask(y).to(self.device)
                preds = torch.softmax(self.model(x).squeeze(0), dim=0)
                preds = (preds > 0.5).float()
                num_correct += (preds[1:,:,:] == y[1:,:,:]).sum()
                num_pixels += torch.numel(preds[1:,:,:])
                dice_score += (2 * (preds[1:,:,:] * y[1:,:,:]).sum()) / ((preds[1:,:,:] + y[1:,:,:]).sum() + 1e-8)

                
                # Dice Score for each class             
                for c in range(1,len(y)):
                  num_correct_c += (preds[c,:,:] == y[c,:,:]).sum()
                  num_pixels_c += torch.numel(preds[c,:,:])
                  dice_score_c_loop[self.classes[c-1]] += (2 * (preds[c,:,:] * y[c,:,:]).sum()) / ((preds[c,:,:] + y[c,:,:]).sum() + 1e-8)
                
        dice_score_c = {key: float(value / len(self.test_loader)) for key, value in dice_score_c_loop.items()}
        print(
            f"Prefomance: accuracy = {num_correct/num_pixels*100:.2f}, DICE score = {dice_score/len(self.test_loader)}, DICE_score_c = {dice_score_c}"
        )
        self.model.train()
        
        return dice_score/len(self.test_loader), dice_score_c
    
    def train(self):

        SCORE = 0.
        counter = 0
        with open(os.path.join(self.out_dir, 'training_dice_score_val.json'), 'a') as f_score_val, \
             open(os.path.join(self.out_dir, 'training_dice_score_ind_c_val.json'), 'a') as f_score_ind_c_val, \
             open(os.path.join(self.out_dir, 'training_dice_score_test.json'), 'a') as f_score_test, \
             open(os.path.join(self.out_dir, 'training_dice_score_ind_c_test.json'), 'a') as f_score_ind_c_test, \
             open(os.path.join(self.out_dir, 'training_train_loss.json'), 'a') as f_loss:

          for epoch in range(self.num_epochs):
            counter += 1

            print(f"On epoch {epoch+1}/{self.num_epochs}:")
            train_loss = self.train_fn(epoch)

            # check accuracy
            score, score_c = self._check_accuracy()
            score_test, score_c_test = self._check_accuracy_test()
            
            # Save dice score for val for each epoch
            json.dump(score.detach().cpu().numpy().tolist(), f_score_val)
            f_score_val.write('\n')  # Add a newline to separate entries in the file

            # Save dice score for each class in val for each epoch
            json.dump(score_c, f_score_ind_c_val)
            f_score_ind_c_val.write('\n')  # Add a newline to separate entries in the file

            # Save dice score for test for each epoch
            json.dump(score_test.detach().cpu().numpy().tolist(), f_score_test)
            f_score_test.write('\n') 

             # Save dice score for each class in test for each epoch
            json.dump(score_c_test, f_score_ind_c_test)
            f_score_ind_c_test.write('\n')

               

            # Save train loss for each epoch
            json.dump(train_loss, f_loss)
            f_loss.write('\n')  

            if score > SCORE:
                counter = 0
                SCORE = score
                checkpoint = {
                    "state_dict": self.model.state_dict(),
                    "optimizer":  self.opt.state_dict(),
                    "DICE_score_val": SCORE,
                    "DICE_score_ind_c_val": score_c,
                    "DICE_score_test": score_test,
                    "DICE_score_ind_c_test": score_c_test,
                    "train_loss": train_loss
                }
                
                self._save_predictions_as_imgs(self.test_loader, is_test=True)
                self._save_checkpoint(checkpoint)
            
            if counter == self.patience: break

   
    
    
class Evaluator:
    def __init__(
        self, 
        model_dir,
        classes,
        encoder   = 'efficientnet-b0',
        decoder   = 'Unet',
        ckpt_name = 'training_checkpoint.pth', 
        device    = 'cuda'
    ):
        self.model   = self._load_model(model_dir, encoder, decoder, ckpt_name, len(classes)+1, device)
        self.device  = device
        self.masks   = {}
        self.classes = classes
        self.colors  = get_class_colors(classes)
        self.model_dir = model_dir
    
    def _load_model(self, path, encoder, decoder, ckpt_name, num_classes, device):
        ckpt = torch.load(os.path.join(path, ckpt_name))
        
        if decoder == 'Unet':
            model = smp.Unet(
                        encoder_name    = encoder,     
                        encoder_weights = None,  
                        in_channels     = 1,               
                        classes         = num_classes,                   
                    )
            
        elif decoder == 'UnetPlusPlus':
            model = smp.UnetPlusPlus(
                        encoder_name    = encoder,     
                        encoder_weights = None,  
                        in_channels     = 1,               
                        classes         = num_classes,                   
                    )
        
        model.to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        
        return model
    
    def _get_loader(self, path):
        
        ds = MyDataset(
              dataset_dir = path,
              transform   = get_val_transforms(),
             )


        return DataLoader(
                          ds,
                          batch_size  = 1,
                          num_workers = 0,
                          pin_memory  = False,
                          shuffle     = False,
                         )
        
    def evaluate(self, dataset_dir, dataset_name, conf = 0.5, visualize=True):
        loader = self._get_loader(os.path.join(dataset_dir, dataset_name))
        
        for batch_idx, (x, _, name) in enumerate(loader):
          print(name)
          x = x.to(device=self.device)
          with torch.no_grad():
              preds = torch.softmax(self.model(x.float()).squeeze(0),dim = 0)
          preds = (preds>0.5).cpu().numpy()

          if visualize:
            os.makedirs(os.path.join(dataset_dir,dataset_name,'pred',self.model_dir.split("/dump/")[1]), exist_ok=True)
            self._save_single_pred(dataset_dir, dataset_name, preds, name)


    def _save_single_pred(self, dataset_dir, dataset_name, preds, name):
      rgb = cv2.imread(os.path.join(dataset_dir, dataset_name,'images', name[0]), cv2.IMREAD_COLOR) 
      rgb = A.Compose(
          [
              A.Resize(IMGSZ, IMGSZ, interpolation=cv2.INTER_NEAREST, always_apply=True, p=1),
          ],
      )(image=rgb)['image']


      for ch in range(3):
          for idx, cls_name in enumerate(self.classes): 
              rgb[:, :, ch][preds[idx + 1] != 0] = self.colors[cls_name][ch]

      # Get contours for class FP
      contours_fp, _ = cv2.findContours((preds[1] == 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      # Draw arrow from center of FP towards FPD if FPD is present
      if  np.max(preds[2]) != 0:
          # Get center of FP
          cords_FP = np.where(preds[1]==1)
          cy_fp,cx_fp = int(np.mean(cords_FP[0])),int(np.mean(cords_FP[1]))

          # Get center of FPD
          cords_FPD = np.where(preds[2]==1)
          cy_fpd,cx_fpd = int(np.mean(cords_FPD[0])),int(np.mean(cords_FPD[1]))

          # Calculate the vector from center of FP to center of FPD
          vector_fp_fpd = np.array([cx_fpd - cx_fp, cy_fpd - cy_fp])

          # Scale the vector to make the arrow longer (e.g., multiplying by 1.2)
          scaled_vector = 1.1 * vector_fp_fpd

          # Calculate the endpoint of the arrow after scaling
          endpoint = (cx_fp + scaled_vector[0], cy_fp + scaled_vector[1])

          # Draw small circle at the beginning of the arrow
          cv2.circle(rgb, (cx_fp, cy_fp), radius=10, color=[0, 0, 255])  # Blue color for the circle

          # Draw arrow
          arrow_color = [0, 0, 255]  # Blue color for the arrow
          rgb = cv2.arrowedLine(rgb, (cx_fp, cy_fp), (int(endpoint[0]), int(endpoint[1])), arrow_color, thickness=2)

      plt.imshow(rgb)
      plt.show()
      skimage.io.imsave(os.path.join(dataset_dir, dataset_name,'pred', self.model_dir.split("/dump/")[1], name[0]), rgb)


    
        
        
        
        