import pygame
import csv
import time

# Initialize pygame
pygame.init()

# Game setup
WIDTH, HEIGHT = 800, 400
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Movement Logger - Ayaan")

clock = pygame.time.Clock()

# Player variables
player = pygame.Rect(100, 300, 50, 50)
velocity = 5
jump = False
jump_height = 10
jump_count = jump_height

# Logging
fieldnames = ["timestamp", "x_position", "y_position", "action"]
log_data = []

running = True
start_time = time.time()

while running:
    clock.tick(30)
    win.fill((255, 255, 255))

    keys = pygame.key.get_pressed()
    action = "idle"

    if keys[pygame.K_LEFT]:
        player.x -= velocity
        action = "left"
    if keys[pygame.K_RIGHT]:
        player.x += velocity
        action = "right"
    if not jump and keys[pygame.K_SPACE]:
        jump = True
        action = "jump"

    if jump:
        if jump_count >= -jump_height:
            neg = 1 if jump_count > 0 else -1
            player.y -= (jump_count ** 2) * 0.4 * neg
            jump_count -= 1
        else:
            jump = False
            jump_count = jump_height

    # Draw player
    pygame.draw.rect(win, (0, 0, 255), player)

    # Log data
    timestamp = round(time.time() - start_time, 2)
    log_data.append({
        "timestamp": timestamp,
        "x_position": player.x,
        "y_position": player.y,
        "action": action
    })

    # Display update
    pygame.display.update()

    # Event quit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

# Save data
with open("movement_data.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(log_data)

pygame.quit()
print("✅ Movement logged and saved to movement_data.csv")
