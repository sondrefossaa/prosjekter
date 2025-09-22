import cv2
import numpy as np
from scrython import Search
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from PIL import Image
import matplotlib.pyplot as plt
import scrython
from Levenshtein import distance
from difflib import SequenceMatcher
import requests
import json

llanowar_elves = r"test images\PXL_20250919_074527542.jpg"
khalni = r"test images\PXL_20250919_074534643.MP.jpg"
voyaging_sartyr = r"test images\PXL_20250919_074530166.jpg"
arbor_elf = r"test images\PXL_20250919_074532601.jpg"

import requests
import time

def get_all_card_names_efficient():
    """
    More efficient method using Scryfall's pagination
    """
    card_names = set()  # Use set to avoid duplicates
    next_page = "https://api.scryfall.com/cards/search?q=game:paper"
    
    while next_page:
        time.sleep(0.1)  # Respect rate limits
        
        response = requests.get(next_page)
        
        if response.status_code == 200:
            data = response.json()
            
            for card in data.get('data', []):
                card_names.add(card['name'])
            
            next_page = data.get('next_page')
            print(f"Fetched page, total names: {len(card_names)}")
            
        else:
            print(f"Error: {response.status_code}")
            break
    
    return sorted(list(card_names))

class ImagePreprocessor():
    def __init__(self):
        self.blur_ksize = (5, 5)
        self.canny_threshold_low = 50
        self.canny_threshold_high = 150
        self.min_contour_area = 5000
        self.contrast_alpha = 1.5
        self.contrast_beta = 40
        self.card_width = 488
        self.card_height = 680
    def display_image(self, img, title="Image"):
        """Display an image using matplotlib"""
        plt.figure(figsize=(10, 8))
        if len(img.shape) == 2:  # Grayscale image
            plt.imshow(img, cmap='gray')
        else:  # Color image
            #Convert from BGR to RGB
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        plt.show()
    def load_image(self, path):
        """Load image from path"""
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Image not found at path: {path}")
        return image
    def resize_image(self, image, max_dimension = 1000):
        """Resize the image while maintaining aspect ratio"""
        height, width = image.shape[:2]
        
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        return image
    def adjust_brightness_contrast(self, image):
        """Adjust brightness and contrast of the image"""
        adjusted = cv2.convertScaleAbs(image, alpha=self.contrast_alpha, beta = self.contrast_beta)
        return adjusted
    def preprocess_for_contour_detection(self,image):
        """
        Prepare image for edge detection by:
        1. Converting to grayscale
        2. Applying Guassian blur
        3. Detecting edges
        """
        
        #Make image grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        #Apply guassian blur
        blurred = cv2.GaussianBlur(gray, self.blur_ksize, 0)
        
        #Detect edges using canny edge detector
        edges = cv2.Canny(blurred, self.canny_threshold_low, self.canny_threshold_high)
        
        #Dialate edges to close gaps
        kernel = np.ones((3, 3), np.uint8)
        dialated = cv2.dilate(edges, kernel, iterations=2)
        
        return dialated
    
    def find_card_fom_incomplete_text_contours(self, edges):
        """
        Find card contous inthe edge image and specify potential card contours
        """
        #Find the edge map
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        card_contours = []
        
        for contour in contours:
            #Calculate contour area
            area = cv2.contourArea(contour)
            
            #Skip small contour areas
            if area < self.min_contour_area:
                continue
            
            #Calculate perimiter
            perimeter = cv2.arcLength(contour, True)
            
            #Approximate the contour to reduce the nomber of points
            epsilon = 0.02 * perimeter # Approximation accuracy
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            #Check if approximated contoru has 4 points (quadrilateral)
            if len(approx) == 4:
                #Check if contour is convex (like a card) the angles dont dent inward
                if cv2.isContourConvex(approx):
                    card_contours.append(approx)
                    
            #Sort by largest area
            card_contours = sorted(card_contours, key=cv2.contourArea, reverse=True)
                
            return card_contours
    def extract_title(self, contours):
        pass

    def order_points(self, points):
        """
        Arrange points in consistent order:
        top-left, top-right, bottom-right, bottom-left
        """
        #Initialize a list of coordinates that will be ordered
        rect = np.zeros((4,2), dtype="float32")
        
        #The top-left will have the smallest sum while the bottom-right will have the largest sum
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)] #top-left
        rect[2] = points[np.argmax(s)] #bottom-right
        
        #The top-right will have the smallest difference while the bottom-left will have the largest difference
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)] 
        rect[3] = points[np.argmax(diff)]
        
        return rect
    
    def do_four_point_transform(self, image, points):
        """
        Apply perspective transform to get a birds eye view of the card
        """
        # Get a consistent order of points
        rect = self.order_points(points)
        (tl, tr, br, bl) = rect
        
        # Compute the width of the new image
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        # Compute the height of the new image
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        # Define destination points
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]], dtype="float32")

        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(rect, dst)

        # Apply the perspective transformation
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        resiezd = cv2.resize(warped, (self.card_width, self.card_height))
        return resiezd
    def preprocess_for_ocr(self, card_image):
        """
        Prepare the extracted card image for OCR text recognition
        """
        # Convert to grayscale
        gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to binarize the image
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply slight dilation to make text more connected
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.dilate(binary, kernel, iterations=1)
        
        return processed
    def process_image(self, image_path, debug=False):
        """
        Main method to process an image and extract card
        """
        # Load and resize image
        image = self.load_image(image_path)
        image = self.resize_image(image)
        
        if debug:
            self.display_image(image, "Original Image")
        
        # Adjust brightness and contrast
        enhanced = self.adjust_brightness_contrast(image)
        
        if debug:
            self.display_image(enhanced, "Enhanced Image")
        
        # Preprocess for contour detection
        edges = self.preprocess_for_contour_detection(enhanced)
        
        if debug:
            self.display_image(edges, "Edge Detection")
        
        # Find card contours
        card_contours = self.find_card_fom_incomplete_text_contours(edges)
        
        
        if not card_contours:
            print("No card contours found")
            return None, None
        
        # Draw contours on original image for visualization
        contour_image = image.copy()
        cv2.drawContours(contour_image, card_contours, -1, (0, 255, 0), 3)
        
        if debug:
            self.display_image(contour_image, "Detected Contours")
        
        # Extract the first (largest) card
        card_contour = card_contours[0]
        # Apply perspective transform
        extracted_card = self.do_four_point_transform(image, card_contour.reshape(4, 2))
        
        if debug:
            self.display_image(extracted_card, "Extracted Card")
        
        # Preprocess for OCR
        ocr_ready = self.preprocess_for_ocr(extracted_card)
        
        if debug:
            self.display_image(ocr_ready, "OCR Ready")
        
        return extracted_card, ocr_ready

#Pytesseract
class ocr_process():
    def __init__(self):
        self.name_height = 0.15
        self.name_width = 0.7
    
    def extract_title_img(self, ocr_image):
        img_h, img_w, _ = ocr_image.shape

        crop_h = int(img_h * self.name_height)
        crop_w = int(img_w * self.name_width)

        x, y = 0, 0  # for example, top-left
        cropped = ocr_image[y:y+crop_h, x:x+crop_w]

        return cropped
    def get_text(self, image, debug = False):
        title_cropped = self.extract_title_img(image)

        text = pytesseract.image_to_string(title_cropped, config="--psm 7")  # PSM 7 = single line of text

        if debug: (print("Detected title:", text.strip()))
        return text

    def find_card_fom_incomplete_text(self, image):
        #card_names = get_all_card_names_efficient()
        #print(card_names[0:10])
        with open("Scryfall cards normal image.json") as f:
            cardjson = json.load(f)

        found_text = self.get_text(image)
        
        levensteins_max = 0
        levensteins_id = 0
        i = 0
        #Maybe rewrite in c++
        for card in cardjson:
            matcher  = SequenceMatcher(None, card["name"], found_text)
            dist = matcher.ratio()
            if dist > levensteins_max:
                levensteins_max = dist
                levensteins_id = i
            i += 1
        
        return cardjson[levensteins_id]

    def find_info(self, image, type):
        return self.find_card_fom_incomplete_text(image)[type]
    
    def extract_set(self, image):
        img_h, img_w, _ = image.shape

        #crop_h = int(img_h * self.name_height)
        #crop_w = int(img_w * self.name_width)
        #TODO Crop with edge detection
        x, y = img_w * 0.8, img_h/2  # for example, top-left
        x = int(x)
        y = int(y)
        #crop in
        cropped = image[y:y+100, x:x+100]
        self.display(cropped)

        #Apply B/W
        gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
        _, BW_cropped = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        self.display(BW_cropped)
        return cropped
    def find_set_from_image(self, image):
        pass


    def display(self, image):
        """ imagejson = self.find_card_fom_incomplete_text(image)
        imageurl = imagejson["image_url"]
        bilde = requests.get(imageurl)
        bilde.raise_for_status()

        img_array = np.frombuffer(bilde.content, np.uint8)
        im = cv2.imdecode(img_array, cv2.IMREAD_COLOR_BGR)
        if not im: return
        img_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) """

        cv2.imshow("Card", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
# Example usage and debugging
if __name__ == "__main__":
    # Create preprocessor
    preprocessor = ImagePreprocessor()
    ocrpros = ocr_process()
    # Process an image with debugging enabled
    #card_image, ocr_image = preprocessor.process_image(card2_path, debug=True)
    ocr_image = preprocessor.process_image(khalni, True)[0]
    #title = ocrpros.extract_title_img(ocr_image)
    #preprocessor.display_image(title)
    #ocrpros.display(ocr_image)
    print(ocrpros.find_info(ocr_image, "name"))
    #ocrpros.extract_set(ocr_image)


    """ if card_image is not None:
        # Save the processed images
        cv2.imwrite("extracted_card.jpg", card_image)
        cv2.imwrite("ocr_ready.jpg", ocr_image)
        print("Card extracted and processed successfully!") """