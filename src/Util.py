"""
Author: Mehmet Kayra Oguz and Chat GPT
Date: July 10, 2023
"""

def save_file(content:str, path='output.txt'):
    with open(path, 'w') as file:
        file.write(content)
        
# GPT
def split_level_into_rectangles(file_path, width, height):
    # Read the contents of the file
    with open(file_path, 'r') as file:
        content = file.read()

    # Split the content into lines
    lines = content.split('\n')

    # Remove empty lines
    lines = [line for line in lines if line]

    # Calculate the number of complete rectangles that can fit
    num_rows = len(lines) // height
    num_cols = len(lines[0]) // width

    rectangles = []

    # Iterate over each complete rectangle
    for row in range(num_rows):
        for col in range(num_cols):
            # Calculate the starting and ending indices for the current rectangle
            start_row = row * height
            end_row = start_row + height
            start_col = col * width
            end_col = start_col + width

            # Extract the rectangle from the lines
            rectangle = '\n'.join(line[start_col:end_col]
                                  for line in lines[start_row:end_row])
            rectangles.append(rectangle)

    # Pack the remaining characters into the last rectangle
    remaining_rows = len(lines) % height
    remaining_cols = len(lines[0]) % width

    if remaining_rows > 0 or remaining_cols > 0:
        last_rectangle = ''

        # Add the remaining rows
        if remaining_rows > 0:
            last_rectangle += '\n'.join(line[-remaining_cols:]
                                        for line in lines[-remaining_rows:])

        # Add the remaining columns
        if remaining_cols > 0:
            last_rectangle += '\n'.join(line[-remaining_cols:]
                                        for line in lines[-height:])

        rectangles.append(last_rectangle)

    return rectangles

# GPT
def combine_squares_into_level(outputs, with_ruler=False):
    # Initialize a list to store each line of the final string
    combined = []

    # Split each square into lines
    split_squares = [square.split('\n') for square in outputs]

    # Assume all squares have the same height
    square_height = len(split_squares[0])

    # Initialize the combined list with the same number of items as the height of a square
    for _ in range(square_height):
        combined.append('')

    # Iterate over each line in each square
    for square in split_squares:
        for i in range(square_height):
            # Append each line of the square to the corresponding line in the combined string
            combined[i] += square[i]

    # Join the lines into a single string
    content = '\n'.join(combined)
    
    if with_ruler:
        ruler = "\n"
        for i in range((len(outputs)+1)*16):
            if i % 16 == 0:
                ruler += "|"
            else:
                ruler += " "
        content += ruler
    return content