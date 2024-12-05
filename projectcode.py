import os
import cv2
import easyocr
import numpy as np
import imutils
import smtplib
import time
# Global variables
email = "jeevaj3v12@gmail.com"
recipient = "jeevaj3v12@gmail.com"
subject = "INTRUDER ALERT"
message = "Someone parked their vehicle who doesn't have eligibility to park here. Please take necessary action."

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Path to the folder containing images for training
training_folder_path = 'C:/Users/JESWIN/OneDrive/Documents/python/opencv projects/myvenv/Scripts/proj/Images'

# Path to the folder containing images with access number plates
access_folder_path = 'C:/Users/JESWIN/OneDrive/Documents/python/opencv projects/myvenv/Scripts/proj/Images'

# Create an EasyOCR reader object
reader = easyocr.Reader(['en'], gpu=True)

# Function to train and understand license plates from images
def train_and_understand():
    # Get a list of all files in the training folder
    training_files = os.listdir(training_folder_path)

    # Iterate through each image file in the training folderq
    for file in training_files:
        image_path = os.path.join(training_folder_path, file)
        image = cv2.imread(image_path)

        # Preprocessing: Convert to grayscale, apply Gaussian blur, and enhance contrast
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        enhanced_image = cv2.equalizeHist(blurred_image)

        # Preprocessing: Convert to grayscale and apply adaptive thresholding
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray_image, 11, 17, 17) #Noise reduction
        edged = cv2.Canny(bfilter, 30, 200) #Edge detection


        # Preprocessing: Convert to grayscale, apply Gaussian blur, and enhance contrast


        # Find contours in the edged image
        contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        # Find the largest contour (assuming the license plate is the largest)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            # Create a rectangle surrounding the contour
            x, y, w, h = cv2.boundingRect(contour)
            # Create a mask with the same shape as the gray_image
            mask = np.zeros(gray_image.shape, dtype=np.uint8)
            # Draw the contour on the mask
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
            # Crop the region of interest from the gray_image
            cropped_image = gray_image[y:y+h, x:x+w]
        else:
            print("No contours found.")
            continue

        # Perform OCR on the cropped image
        results = reader.readtext(cropped_image)

        # Print the extracted text (ignoring "IND")
        for result in results:
            if "IND" not in result[1]:
                print(f"License plate text from {file}: {result[1]}")


# Function to compare text from captured image with texts from images in the access folder
def check_access(text):
    access_files = os.listdir(access_folder_path)
    for file in access_files:
        image_path = os.path.join(access_folder_path, file)
        image = cv2.imread(image_path)
        image_np = np.array(image)
        results = reader.readtext(image_np)
        for result in results:
            if text == result[1]:
                return True
    return False

# Function to capture video from camera and check access
def capture_video_and_check_access():
    # Initialize camera
    camera = cv2.VideoCapture(0)
    time.sleep(5)


    while True:
        # Capture frame-by-frame
        ret, frame = camera.read()

        # Convert frame to a NumPy array
        frame_np = np.array(frame)

        # Perform OCR on the captured frame
        results = reader.readtext(frame_np)

        # Extract text from OCR results
        extracted_text = [result[1] for result in results]

        print("Recognized Text:")
        for text in extracted_text:
            print(text)

        # Check if extracted text matches with any of the texts in access folder
        for text in extracted_text:
            if check_access(text):
                print("Access granted.")
                camera.release()
                return
            else:
                print("Access denied. Sending email notification.")
                send_email() # Implement email notification
                camera.release()
                return

        # Display the captured frame
        cv2.imshow('Live Video Feed', frame)

        # Check for 'q' key press to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release camera
    camera.release()
    cv2.destroyAllWindows()


def send_email():
    global email, recipient, subject, message

    # Construct the email message
    text = f"Subject:{subject}\n\n{message}"

    # Connect to SMTP server
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()

    try:
        # Login to the email account
        server.login(email, "gnfm czrk hkjx dzbg")
       
        # Send email
        server.sendmail(email, recipient, text)
        print("Email has been sent to " + recipient)
    except Exception as e:
        print("An error occurred while sending the email:", str(e))
    finally:
        # Close the connection
        server.quit()

# Main function
def main():
    # Train and understand license plates from images in the training folder
    train_and_understand()

    # Capture video from camera and check access
    capture_video_and_check_access()

if __name__ == "__main__":
    main()
 
