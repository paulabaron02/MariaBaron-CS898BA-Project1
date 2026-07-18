"""
Homework One
Maria Paula Baron Rodriguez 
Wsu ID: J858Q278
"""

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import scipy.stats as sp
import random
import shutil
import re


plt.close('all')

Results = r"C:\Users\Paula\.spyder-py3\MariaBaron-CS898BA-Project1\MariaBaron-CS898BA-Project1\Results"

if os.path.exists(Results):
    shutil.rmtree(Results)
os.makedirs(Results)

#load the image
Image_Alien = cv2.imread(r"C:\Users\Paula\.spyder-py3\MariaBaron-CS898BA-Project1\MariaBaron-CS898BA-Project1\HW1_IMG_CS898BA.png")
Image_Alien_RGB = cv2.cvtColor(Image_Alien, cv2.COLOR_BGR2RGB)
plt.imshow(Image_Alien_RGB)
plt.title("Original")
plt.axis("off")
plt.tight_layout()
plt.show()
plt.savefig("Results/Original.png")

# %%
#2.1 Find and print basic image statistics of the original image for each 
#individual channel (min, max, average, median, mode, skew, range, standard
# deviation, variance)

BlueChannel, GreenChannel, RedChannel = cv2.split(Image_Alien)
Statics_BlueChannel = {
    "Min": BlueChannel.min(),
    "Max": BlueChannel.max(),
    "Average": BlueChannel.mean(),
    "Median": np.median(BlueChannel),
    "Skew": sp.skew(BlueChannel.flatten()),
    "Range": (BlueChannel.max() - BlueChannel.min()),
    "Standard Deviation": (BlueChannel.std()),
    "Variance": (BlueChannel.var()),
}

modo = sp.mode(BlueChannel.flatten(), keepdims=False)
Statics_BlueChannel["Mode"] = int(modo.mode)

#creates a full image with the values and the respective image in the color 
df = pd.DataFrame(Statics_BlueChannel.items(),columns=["Statics","value"])
fig, axes = plt.subplots(3,2)
axes = axes.flatten()
ax1, ax2, ax3, ax4, ax5, ax6 = axes
ax1.imshow(BlueChannel, cmap='gray')
ax1.axis('off') 
ax1.set_title('Blue Channel', fontsize=9)
tabla = ax2.table(cellText=df.values, 
                  colLabels=df.columns, 
                  cellLoc='center', 
                  loc='center')
tabla.scale(0.85, 0.7) 
ax2.axis('off') 
ax2.set_title('Blue Channel Statics', fontsize=9)
plt.tight_layout()


# %%
Statics_RedChannel = {
    "Min": RedChannel.min(),
    "Max": RedChannel.max(),
    "Average": RedChannel.mean(),
    "Median": np.median(RedChannel),
    "Skew": sp.skew(RedChannel.flatten()),
    "Range": (RedChannel.max() - RedChannel.min()),
    "Standard Deviation": (RedChannel.std()),
    "Variance": (RedChannel.var()),
}
modo = sp.mode(RedChannel.flatten(), keepdims=False)
Statics_RedChannel["Mode"] = int(modo.mode)

#creates a full image with the values and the respective image in the color 
df1 = pd.DataFrame(Statics_RedChannel.items(),columns=["Statics","value"])

ax3.imshow(RedChannel, cmap='gray')
ax3.axis('off') 
ax3.set_title('Red Channel', fontsize=9)
tabla1 = ax4.table(cellText=df1.values, 
                  colLabels=df1.columns, 
                  cellLoc='center', 
                  loc='center')
tabla1.scale(0.85, 0.7) 
ax4.axis('off') 
ax4.set_title('Red Channel Statics', fontsize=9)
plt.tight_layout()

# %%
Statics_GreenChannel = {
    "Min": GreenChannel.min(),
    "Max": GreenChannel.max(),
    "Average": GreenChannel.mean(),
    "Median": np.median(GreenChannel),
    "Skew": sp.skew(GreenChannel.flatten()),
    "Range": (GreenChannel.max() - GreenChannel.min()),
    "Standard Deviation": (GreenChannel.std()),
    "Variance": (GreenChannel.var()),
}
modo = sp.mode(GreenChannel.flatten(), keepdims=False)
Statics_GreenChannel["Mode"] = int(modo.mode)

#creates a full image with the values and the respective image in the color 
df2 = pd.DataFrame(Statics_GreenChannel.items(),columns=["Statics","value"])
ax5.imshow(GreenChannel, cmap='gray')
ax5.axis('off') 
ax5.set_title('Green Channel', fontsize=9)
tabla2 = ax6.table(cellText=df2.values, 
                  colLabels=df2.columns, 
                  cellLoc='center',
                  loc='center')
tabla2.scale(0.85, 0.7) 
ax6.axis('off') 
ax6.set_title('Green Channel Statics', fontsize=9)
plt.tight_layout()
print("2.1 Complete")

# %%
#2.2 Convert and save the image to greyscale, binary, and different color 
#spaces (HSV, CIELAB, and HLS).


Grey_Image = cv2.cvtColor(Image_Alien, cv2.COLOR_BGR2GRAY)
Fig3 = plt.figure() 
plt.imshow(Grey_Image, cmap='gray')
plt.title("GreyScale")
plt.axis("off")
plt.tight_layout()
plt.savefig("Results/GreyScale.png")

_,Binary_Image = cv2.threshold(Grey_Image, 127, 255, cv2.THRESH_BINARY)
Fig4 = plt.figure() 
plt.imshow(Binary_Image, cmap='gray')
plt.title("Binary")
plt.axis("off")
plt.tight_layout()

plt.savefig("Results/Binary.png")

HSV_Image = cv2.cvtColor(Image_Alien, cv2.COLOR_BGR2HSV)
HSV_Image_plt = cv2.cvtColor(HSV_Image, cv2.COLOR_BGR2RGB)
Fig5 = plt.figure() 
plt.imshow(HSV_Image_plt )
plt.title("HSV")
plt.axis("off")
plt.tight_layout()
plt.savefig("Results/HSV.png")

CIELAB_Image = cv2.cvtColor(Image_Alien, cv2.COLOR_BGR2LAB)
CIELAB_Imag_plt = cv2.cvtColor(CIELAB_Image, cv2.COLOR_BGR2RGB)
Fig6 = plt.figure() 
plt.imshow(CIELAB_Imag_plt)
plt.title("CIELAB")
plt.axis("off")
plt.tight_layout()
plt.savefig("Results/CIELAB.png")

HLS_Image = cv2.cvtColor(Image_Alien, cv2.COLOR_BGR2HLS)
HLS_Image_plt = cv2.cvtColor(HLS_Image, cv2.COLOR_BGR2RGB)
Fig5 = plt.figure() 
plt.imshow(HLS_Image_plt)
plt.title("HLS")
plt.axis("off")
plt.tight_layout()
plt.savefig("Results/HLS.png")
print("2.2 Complete")

# %%
#2.3 On the HSV converted image, normalize the lighting by performing histogram
# equalization across the V (value) channel.

H_Channel, S_Channel, V_Channel = cv2.split(HSV_Image)
V_Equalized = cv2.equalizeHist(V_Channel)

HSV_Normalized = cv2.merge([H_Channel, S_Channel, V_Equalized])
HSV_Normalized_plt = cv2.cvtColor(HSV_Normalized, cv2.COLOR_BGR2RGB)
Fig6 = plt.figure() 
plt.imshow(HSV_Normalized_plt)
plt.title("HSV_Normalized")
plt.axis("off")
plt.tight_layout()
#plt.savefig("Results/HSV_Normalized.png")
print("2.3 Complete")

# %%
#2.4 Convert the normalized image back to RGB and save it.

Normalized_RGB = cv2.cvtColor(HSV_Normalized, cv2.COLOR_HSV2RGB)
Fig7 = plt.figure() 
plt.imshow(Normalized_RGB)
plt.title("HSV_Normalized_RGB")
plt.axis("off")
plt.tight_layout()
plt.savefig("Results/HSV_Normalized_RGB.png")
print("2.4 Complete")

# %%
#2.6. Perform random affine transformations on each image (you should perform 14 
#total transformations - 2 for each image). Affine transformations can be 
#translation, rotation, scaling, or shear as long as each is unique in either 
#transformation type or transformation value (rotate 90 degrees vs rotate 186 
#degrees). No two images should be transformed in the exact same way. Save each
# of those images to new files.

Images_List = [
    ("Original", Image_Alien),
    ("Greyscale", Grey_Image),
    ("Binary", Binary_Image),
    ("HSV", HSV_Image),
    ("CIELAB", CIELAB_Image),
    ("HLS", HLS_Image),
    ("HSV_Normalized_RGB",  cv2.cvtColor(Normalized_RGB, cv2.COLOR_RGB2BGR))
]

for name, image in Images_List:
    height, width = image.shape[:2]

    # Rotation 
    angle = np.random.randint(15, 180)
    scale = round(np.random.uniform(0.8, 1.2), 2)
    Rotation_Matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, scale)
    Rotated_Image = cv2.warpAffine(image, Rotation_Matrix, (width, height))
    rot_filename = f"{name}_Rot{angle}_Scale{scale}_Rotation.png"
    cv2.imwrite(os.path.join(Results, rot_filename), Rotated_Image)

    # Translation
    tx = np.random.randint(-1000, 500)
    ty = np.random.randint(-500, 200)
    Translation_Matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    Translated_Image = cv2.warpAffine(image, Translation_Matrix, (width, height),
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    trans_filename = f"{name}_Tx{tx}_Ty{ty}_Translation.png"
    cv2.imwrite(os.path.join(Results, trans_filename), Translated_Image)
print("2.6 Complete")
# %%
# 2.8. Apply a Gaussian blur to each image using the 
#levels of sigma: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5. 
#Discuss how the level of sigma changes the image. 
#Save each of those images to new files.

Sigma_Levels = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

# Combine the 7 converted images and the 14 affine images

for Image_Gaussian in os.listdir(Results):

    Full_Path = os.path.join(Results, Image_Gaussian)
    
    image = cv2.imread(Full_Path, cv2.IMREAD_UNCHANGED)
    
    image_name = os.path.splitext(os.path.basename(Image_Gaussian))[0]
    
    for sigma in Sigma_Levels:
        
        Blurred_Image = cv2.GaussianBlur(image,(0, 0),sigmaX=sigma,sigmaY=sigma)
        
        cv2.imwrite(os.path.join(Results,  image_name + "_GaussianBlur_Sigma_" + str(sigma) + ".png"),Blurred_Image)
print("2.8 Complete")
# %%
# 3.1 Randomly create 4 equally sized subsets of the images from part 1.
All_Files = os.listdir(Results)
random.shuffle(All_Files)
Subsets = [All_Files[i*42:(i+1)*42] for i in range(4)]
My_Subset = Subsets[0]
print("3.1 Complete")

# %%
# 3.4 Perform these edge detection techniques on that subset:
# a.  Sobel
# b.  Laplacian
# c.  Canny
# d.  Prewitt

Edge_Folder = os.path.join(Results, "Edge_Detection")
if os.path.exists(Edge_Folder):
    shutil.rmtree(Edge_Folder)
os.makedirs(Edge_Folder)

for file in My_Subset:
    img_path = os.path.join(Results, file)
    img = cv2.imread(img_path)
    
    if img is None:
        continue
    
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    

    name = file.replace('.png', '')
    cv2.imwrite(os.path.join(Edge_Folder, f"{name}_Original.png"), img)
    
    # Sobel
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    sobel = np.uint8(np.clip(sobel, 0, 255))
    cv2.imwrite(os.path.join(Edge_Folder, f"{name}_Sobel.png"), sobel)
    
    # Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = np.uint8(np.abs(lap))
    cv2.imwrite(os.path.join(Edge_Folder, f"{name}_Laplacian.png"), lap)
    
    # Canny
    canny = cv2.Canny(gray, 50, 150)
    cv2.imwrite(os.path.join(Edge_Folder, f"{name}_Canny.png"), canny)
    
    # Prewitt
    kernel_x = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=np.float32)
    kernel_y = np.array([[-1,-1,-1],[0,0,0],[1,1,1]], dtype=np.float32)
    prewitt_x = cv2.filter2D(gray.astype(np.float32), -1, kernel_x)
    prewitt_y = cv2.filter2D(gray.astype(np.float32), -1, kernel_y)
    prewitt = np.sqrt(prewitt_x**2 + prewitt_y**2)
    prewitt = np.uint8(np.clip(prewitt, 0, 255))
    cv2.imwrite(os.path.join(Edge_Folder, f"{name}_Prewitt.png"), prewitt)
    
print("3.4 Complete")


# %%
# 3.8 Create 42, 5-image plots of the input image 
# (from the start of part 3) next to the edge-detected images and
# output 6 random plots to add to the readme. Include information on what 
# processing techniques were used on the images.

Plot_Folder = os.path.join(Results, "Plots")
if os.path.exists(Plot_Folder):
    shutil.rmtree(Plot_Folder)
os.makedirs(Plot_Folder)

def parse_pipeline_title(name):
    parts = []

    # Color space base
    color_spaces = ['Greyscale', 'Binary', 'HSV_Normalized_RGB', 'HSV', 'CIELAB', 'HLS', 'Original']
    for cs in color_spaces:
        if name.startswith(cs):
            parts.append(cs)
            name = name[len(cs):]
            break

    # BGR in case
    if parts and parts[0] not in ['Greyscale', 'Binary']:
        parts.append("BGR")

    # Rotation 
    rot_match = re.search(r'_Rot(-?\d+)_Scale([\d.]+)_Rotation', name)
    if rot_match:
        angle = rot_match.group(1)
        scale = rot_match.group(2)
        parts.append(f"Affine(Rot:{angle}°, Scale:{scale})")

    # Translation 
    trans_match = re.search(r'_Tx(-?\d+)_Ty(-?\d+)_Translation', name)
    if trans_match:
        tx = trans_match.group(1)
        ty = trans_match.group(2)
        parts.append(f"Affine(Trans:[{tx},{ty}])")

    # Gaussian Blur sigma
    sigma_match = re.search(r'GaussianBlur_Sigma_([\d.]+)', name)
    if sigma_match:
        parts.append(f"Gaussian Blur(σ:{sigma_match.group(1)})")

    return "\n→ ".join(parts) if parts else name


originals = [f for f in os.listdir(Edge_Folder) if 'Original' in f]
plot_paths = []

for i, orig in enumerate(originals):
    name = orig.replace('_Original.png', '')

    original = cv2.imread(os.path.join(Edge_Folder, orig))
    if original is None:
        continue

    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    sobel   = cv2.imread(os.path.join(Edge_Folder, f"{name}_Sobel.png"), 0)
    lap     = cv2.imread(os.path.join(Edge_Folder, f"{name}_Laplacian.png"), 0)
    canny   = cv2.imread(os.path.join(Edge_Folder, f"{name}_Canny.png"), 0)
    prewitt = cv2.imread(os.path.join(Edge_Folder, f"{name}_Prewitt.png"), 0)

    if any(x is None for x in [sobel, lap, canny, prewitt]):
        continue

    fig = plt.figure(figsize=(10, 9))

    pipeline_title = parse_pipeline_title(name)
    fig.suptitle(
        f"Sample {i+1} Pipeline Trajectory:\n{pipeline_title}",
        fontsize=9, y=0.99
    )

    ax_sobel = fig.add_subplot(3, 3, 2)
    ax_sobel.imshow(sobel, cmap='gray')
    ax_sobel.set_title("Sobel Edge", fontsize=8)
    ax_sobel.axis('off')

    ax_lap = fig.add_subplot(3, 3, 4)
    ax_lap.imshow(lap, cmap='gray')
    ax_lap.set_title("Laplacian Edge", fontsize=8)
    ax_lap.axis('off')

    ax_input = fig.add_subplot(3, 3, 5)
    ax_input.imshow(original_rgb)
    ax_input.set_title("Input Image", fontsize=8)
    ax_input.axis('off')

    ax_canny = fig.add_subplot(3, 3, 6)
    ax_canny.imshow(canny, cmap='gray')
    ax_canny.set_title("Canny Edge", fontsize=8)
    ax_canny.axis('off')

    ax_prewitt = fig.add_subplot(3, 3, 8)
    ax_prewitt.imshow(prewitt, cmap='gray')
    ax_prewitt.set_title("Prewitt Edge", fontsize=8)
    ax_prewitt.axis('off')

    for pos in [1, 3, 7, 9]:
        fig.add_subplot(3, 3, pos).axis('off')

    plt.tight_layout()
    plot_path = os.path.join(Plot_Folder, f'Plot_{i+1:03d}.png')
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
   # plt.close()
  #  plot_paths.append(plot_path)
    
print("3.8 Complete")



