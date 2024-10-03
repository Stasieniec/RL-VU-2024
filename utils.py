#-------------------------------------------------------------------------------------------------------------------------------
# Utils file that encompases functions for rendering and visualizing 
#
# Section 1 - Rendering The Maze Environment
#   1. init_window
#   2. render_frame
#   3. close_window
#
# Section 2 - Visualizing The data
#   ... To be Filled ...
#-------------------------------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------------------------------
# Section 1 - Rendering the Maze Environment
#-------------------------------------------------------------------------------------------------------------------------------
import pygame
import numpy as np


def init_window(window, window_size):
    """Initialize the PyGame window."""
    if window is None:
        pygame.init()
        pygame.display.init()
        window = pygame.display.set_mode((window_size, window_size))
    return window

def render_frame(window, window_size, maze, agent_position, sub_goal_position, end_goal_position):
    """Render the environment on a PyGame window."""
    size = maze.shape[0]  # Assume the maze is a square
    pix_square_size = window_size / size  # The size of each grid square in pixels

    # Initialize the canvas
    canvas = pygame.Surface((window_size, window_size))
    canvas.fill((255, 255, 255))  # White background

    # Draw the walls, open spaces, and goals
    for i in range(size):
        for j in range(size):
            if maze[i, j] == 1:  # Wall
                color = (0, 0, 0)  # Black
            elif maze[i, j] == 0:  # Open space
                color = (255, 255, 255)  # White
            elif maze[i, j] == 2:  # Start position
                color = (0, 255, 0)  # Green
            elif maze[i, j] == 3:  # Sub-goal
                color = (255, 255, 0)  # Yellow
            elif maze[i, j] == 4:  # End-goal
                color = (255, 0, 0)  # Red
            pygame.draw.rect(
                canvas,
                color,
                pygame.Rect(j * pix_square_size, i * pix_square_size, pix_square_size, pix_square_size)
            )

    # Draw the agent
    pygame.draw.circle(
        canvas,
        (0, 0, 255),  # Blue for the agent
        ((agent_position[1] + 0.5) * pix_square_size, (agent_position[0] + 0.5) * pix_square_size),
        pix_square_size / 3
    )

    # Draw the grid lines
    for x in range(size + 1):
        pygame.draw.line(canvas, (0, 0, 0), (0, pix_square_size * x), (window_size, pix_square_size * x), width=3)
        pygame.draw.line(canvas, (0, 0, 0), (pix_square_size * x, 0), (pix_square_size * x, window_size), width=3)

    if window is not None:
        window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

def close_window(window):
    """Close the PyGame window."""
    if window is not None:
        pygame.display.quit()
        pygame.quit()

#-------------------------------------------------------------------------------------------------------------------------------