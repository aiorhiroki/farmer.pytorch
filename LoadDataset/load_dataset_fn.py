def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


# img = np.array(Image.open(img_file).resize((640, 320))) / 255.
# mask = np.array(Image.open(mask_file).resize((640, 320)))
# mask[mask == 209] = 1
# mask[mask == 206] = 1
# mask[mask > 1] = 0
# mask = np.eye(2)[mask]