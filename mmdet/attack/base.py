
class Attacker:
    def __init__(self, model, img_transform=(lambda x:x, lambda x:x)):
        self.model = model  # 必须是pytorch的model
        '''self.model.eval()
        for k, v in self.model.named_parameters():
            v.requires_grad = False'''
        self.img_transform=img_transform
        self.forward = lambda attacker, images, labels: attacker.step(images, labels, attacker.loss)

    def set_para(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k,v)

    def set_forward(self, forward):
        self.forward=forward

    def step(self, images, labels, loss):
        pass

    def set_loss(self, loss):
        self.loss=loss

    def attack(self, images, labels):
        pass


import torch
class mmAttacker:
    def __init__(self, model, eps=0.005):
        self.model = model
        self.eps = eps

    def attack(self, data_batch):
        result, image = self.model(**data_batch)
        loss_cls = result['loss_cls'][0]
        loss_bbox = result['loss_bbox'][0]
        loss = loss_cls+loss_bbox
        # print(image.grad)
        # loss_cls.backward()
        # loss_bbox.backward()
        loss.backward()
        perturbation = self.eps*255*torch.sign(image.grad)
        adv_image = image+perturbation
        return adv_image

    
class Noisier:
    def __init__(self, model,std=10.2):
        self.model = model
        self.std = std

    def attack(self, data_batch):
        image = data_batch['img']._data[0]
        shape = image.shape
        perturbation = torch.normal(0.0,self.std,shape)
        adv_image = image+perturbation
        return adv_image







