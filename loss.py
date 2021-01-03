import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):

    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        # S is split size of image
        # B is number of boxes
        # C is number of classes

        self.S = S
        self.B = B
        self.C = C

        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # Predictions need to be reshaped (BATCH_SIZE, S*S(C+B*5)
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        # 0-19 = class probabilities, 20 = class score, 21-25 = bounding box values for 1st box

        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        # Bounding box values for 2nd box, target is the same

        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, bestbox = torch.max(ious, dim=0)
        # bestbox = argmax 

        exists_box = target[..., 20].unsqueeze(3)
        # Will be 0 or 1 depending what is in box, identity of object i

        # Box coordinates(mid point, width, height)
        box_predictions = exists_box * ((
                bestbox * predictions[..., 26:30] # 1 if the second bb is best
                + (1 - bestbox) * predictions[..., 21:25] # for the other
            ))

        box_targets = exists_box * target[..., 21:25]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt( torch.abs(box_predictions[..., 2:4] + 1e-6))
        # Abs - we cant have negative prediction, could happen in initialization (sqrt error)
        # Problem of 0 derivative solved by adding small constant (no inf)

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # Decrease the dimensionality, for MSEs
        box_loss = self.mse(torch.flatten(box_predictions, end_dim=-2),
                            torch.flatten(box_targets, end_dim=-2),)

        # Object loss

        pred_box = (bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21])
        
        # Number of examples for all cells, for each example (N*S*S, 1)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        # No object loss
        # N, S, S, 1 -> N, S*Ss

        no_object_loss = self.mse(torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
                                  torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),)

        no_object_loss += self.mse(torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
                                   torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1))

        # Class loss
        # Dimensionality reduction, each cell is a seperate example

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2,),
            torch.flatten(exists_box * target[..., :20], end_dim=-2,),
        )

        loss = (self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss) # Loss in paper

        return loss