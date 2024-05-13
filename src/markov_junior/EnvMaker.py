"""
Author: Mehmet Kayra Oguz
Date: July 10, 2023
"""

def make_env(width, height, color, path='../resources/environment.txt'):
    env = ""
    for h in range(height):
        for w in range(width):
            env += color
        env += "\n"
    with open(path, 'w') as file:
        file.truncate(0)
        file.write(env)
