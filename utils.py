from PIL import Image, ImageDraw, ImageFont

class Renderer:
    def __init__(self, preprocessor, fontpath='unifont.ttf'):
        # load the font
        self.font = ImageFont.truetype(fontpath, 32)
        self.preprocessor = preprocessor

    def render_image(self, corrupted, original):
        # create a new image with height slightly larger than the font size
        text_length_px = self.font.getsize(corrupted + '  ' + original)[0]
        img = Image.new('RGB', (text_length_px + 20, 40), color='white')
        # create a drawing context
        draw = ImageDraw.Draw(img)
        # draw the text
        draw.text((10, 0), corrupted + '  ' +
                    original, font=self.font, fill='black')
        # return the image
        # return self.preprocessor(img, return_tensors="pt").pixel_values
        return img
