class DataObject(object):

    def __init__(self, case, image, img_spacing, mask, mask_spacing):
        self.case = case
        self.image = image
        self.img_spacing = img_spacing
        self.mask = mask
        self.mask_spacing = mask_spacing

    def __str__(self):
        return self.case
