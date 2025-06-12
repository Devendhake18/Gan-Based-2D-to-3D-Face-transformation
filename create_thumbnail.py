from PIL import Image, ImageDraw, ImageFont
import os

# Create directory if it doesn't exist
assets_dir = os.path.join('frontend', 'public', 'assets')
os.makedirs(assets_dir, exist_ok=True)

# Create a thumbnail image
width, height = 1280, 720
img = Image.new('RGB', (width, height), color=(30, 40, 80))  # Dark blue background
draw = ImageDraw.Draw(img)

# Draw gradient background
for y in range(height):
    r = int(30 + (y / height) * 10)
    g = int(40 + (y / height) * 15)
    b = int(80 + (y / height) * 40)
    for x in range(width):
        draw.point((x, y), fill=(r, g, b))

# Draw overlay rectangle for title
overlay_rect = [(100, 100), (width - 100, 300)]
draw.rectangle(overlay_rect, fill=(0, 0, 0, 128))

# Add title text
title_text = "3D Face Model Tutorial"
try:
    # Try to load a font, use default if not available
    font = ImageFont.truetype("arial.ttf", 64)
    subtitle_font = ImageFont.truetype("arial.ttf", 32)
except:
    # Fall back to default font
    font = ImageFont.load_default()
    subtitle_font = ImageFont.load_default()

draw.text((width//2, 180), title_text, fill=(0, 200, 220), font=font, anchor="mm")
draw.text((width//2, 240), "Learn how to use your generated 3D models", fill=(220, 220, 220), font=subtitle_font, anchor="mm")

# Draw model placeholder
model_rect = [(width - 450, height - 450), (width - 100, height - 100)]
draw.rectangle(model_rect, fill=(20, 30, 60), outline=(0, 200, 220), width=3)
draw.text((width - 275, height - 275), "3D MODEL", fill=(0, 200, 220), font=subtitle_font, anchor="mm")

# Draw feature list
features = [
    "• Import OBJ & MTL files",
    "• Apply textures correctly",
    "• Manipulate 3D models",
    "• Export for various uses"
]

for i, feature in enumerate(features):
    draw.text((200, 350 + i * 60), feature, fill=(255, 255, 255), font=subtitle_font)

# Add play button
play_center = (width//2, height//2 + 50)
play_size = 80
draw.ellipse((play_center[0]-play_size, play_center[1]-play_size, 
              play_center[0]+play_size, play_center[1]+play_size), 
             fill=(0, 188, 212))

# Draw triangle for play icon
play_icon_points = [
    (play_center[0]-30, play_center[1]-40),
    (play_center[0]+50, play_center[1]),
    (play_center[0]-30, play_center[1]+40)
]
draw.polygon(play_icon_points, fill=(255, 255, 255))

# Save the image
thumbnail_path = os.path.join(assets_dir, 'demo-thumbnail.jpg')
img.save(thumbnail_path, quality=90)
print(f"Thumbnail created at: {thumbnail_path}") 