import pygame
import os

# Hide the pygame window
os.environ['SDL_VIDEODRIVER'] = 'dummy'
from PIL import Image

mapsheet_dict = {
'F': 40,
'X': 1,
'#': 2,
'%': 46, #exception
'|': 47, 
'D': 14,
'S': 7,
'C': 7,
'U': 8,
'L': 8,
'o': 15,
't': 18,
'<': 18,
'>': 19,
'[': 20,
']': 21,
'-': 0,
'!': 8,
'Q': 8,
'@': 8,
'?': 8,
'1': 42,
'2': 42,
'g': 48,
'E': 48,
'k': 63, # exception
'M': 49,
'r': 57, # exception
'y': 56,
'*': 54,
'R': 62, # exception
'K': 62, # exception
'T': 58, # exception
}

def world_string_to_png(input_string, root_path, destination_path):


    # Initialize pygame
    pygame.init()

    # Split the input string into rows based on newlines
    rows = input_string.split('\n')
    
    #filter empty rows
    filtered_rows = []
    for element in rows:
        if len(element) != 0:
            filtered_rows.append(element)
    rows = filtered_rows

    # Get world size from string
    world_width_in_blocks=len(rows[0])
    world_height_in_blocks=len(rows)

    # One block
    block_width = 16
    block_height = 16

    # Set the dimensions of the canvas
    canvas_width = world_width_in_blocks * block_width
    canvas_height = world_height_in_blocks * block_height

    # Create the canvas without displaying it
    canvas = pygame.Surface((canvas_width, canvas_height))

    # Load the spritesheet
    spritesheet_image = pygame.image.load(os.path.join(root_path, "src/libraries/mario_level_visualizer/img/mapsheet.png"))

    # Define the size of each sprite
    sprite_width = 16
    sprite_height = 16

    # Create an empty list to store the sprites
    sprites = []

    # Extract individual sprites from the spritesheet
    for y in range(0, spritesheet_image.get_height(), sprite_height):
        for x in range(0, spritesheet_image.get_width(), sprite_width):
            # Create a new sprite surface
            sprite_surface = pygame.Surface((sprite_width, sprite_height), pygame.SRCALPHA)
            sprite_surface.blit(spritesheet_image, (0, 0), pygame.Rect(x, y, sprite_width, sprite_height))

            # Append the sprite surface to the list of sprites
            sprites.append(sprite_surface)

    # Clear the canvas
    canvas.fill((255, 255, 255))

    # Fill background
    for y in range(world_height_in_blocks):
        y_pos = y * sprite_height
        for x in range(world_width_in_blocks):
            x_pos = x * sprite_width
            canvas.blit(sprites[42], (x_pos, y_pos))

    # Draw the sprites
    for y in range(world_height_in_blocks):
        y_pos = y * sprite_height
        for x in range(world_width_in_blocks):
            x_pos = x * sprite_width
            tile_char = rows[y][x]
            sprite_index = mapsheet_dict[tile_char]
            canvas.blit(sprites[sprite_index], (x_pos, y_pos))
            
            # red koopa exception
            if(sprite_index == 57):
                if(y_pos - sprite_height >=0):
                    canvas.blit(sprites[60], (x_pos, y_pos - sprite_height))

            # green koopa exception
            if(sprite_index == 63):
                if(y_pos - sprite_height >=0):
                    canvas.blit(sprites[55], (x_pos, y_pos - sprite_height))
            
            #flag exception
            if(sprite_index == 40):
                for i in range(y-2):
                    canvas.blit(sprites[40], (x_pos, y_pos - i*sprite_height))
                canvas.blit(sprites[41], (x_pos - (sprite_width/2), y_pos - (y-3)*sprite_height))
                canvas.blit(sprites[39], (x_pos, y_pos - (y-2)*sprite_height))

            # winged red koopa exception
            if(sprite_index == 62):
                if(y_pos - sprite_height >0):
                    canvas.blit(sprites[60], (x_pos, y_pos - sprite_height))
                    canvas.blit(sprites[61], (x_pos- (sprite_width/3), y_pos))
            
            # mushroom block exceptions (I know looks horrible but works, so dont touch it. Unless it stops working)
            if(sprite_index == 46):
                canvas.blit(sprites[42], (x_pos, y_pos))
                if(x+1 > world_width_in_blocks-1 and rows[y][x-1]!='%'):
                    canvas.blit(sprites[44], (x_pos, y_pos))
                elif(x-1 < 0 and rows[y][x+1]!='%'):
                    canvas.blit(sprites[45], (x_pos, y_pos))
                elif(not(x+1 > world_width_in_blocks-1 or x-1 < 0)):
                    if(rows[y][x+1]=='%' and rows[y][x-1]=='%'):
                        canvas.blit(sprites[46], (x_pos, y_pos))
                    elif(rows[y][x+1]=='%'):
                        canvas.blit(sprites[44], (x_pos, y_pos))
                    elif(rows[y][x-1]=='%'):
                        canvas.blit(sprites[45], (x_pos, y_pos))
                    else:
                        canvas.blit(sprites[43], (x_pos, y_pos))
                else:
                    canvas.blit(sprites[46], (x_pos, y_pos))

            
            # Piranha plant exception
            if(sprite_index == 58):
                canvas.blit(sprites[42], (x_pos, y_pos))
                if(x+1 >= world_width_in_blocks or y+1 >= world_height_in_blocks):
                    canvas.blit(sprites[42], (x_pos, y_pos))
                elif(rows[y][x+1]!='T'):
                    if(rows[y+1][x]=='<'):
                        canvas.blit(sprites[58], (x_pos+block_width/2, y_pos+5))
                        canvas.blit(sprites[59], (x_pos+block_width/2, y_pos+5-block_height))
                    elif(rows[y+1][x]=='>'):
                        canvas.blit(sprites[58], (x_pos-block_width/2, y_pos+5))
                        canvas.blit(sprites[59], (x_pos-block_width/2, y_pos+5-block_height))
                    else:
                        canvas.blit(sprites[58], (x_pos, y_pos))

    # Save the canvas as an image
    pygame.image.save(canvas, destination_path)
    return Image.open(destination_path)