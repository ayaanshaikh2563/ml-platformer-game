# CATCH ME IF YOU CAN - ML Platformer Game

import pygame
import random
import tensorflow as tf
import numpy as np
import os

# --- Model Loading and Scaler Setup ---
model_path = "enemy_lstm_model.keras"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at: {model_path}. "
                            f"Please run 'train_enemy_lstm.py' first to train and save the LSTM model.")
model = tf.keras.models.load_model(model_path)

# Load scaler min/max values used during training
try:
    scaler_min = np.load("scaler_min.npy")
    scaler_max = np.load("scaler_max.npy")
    num_features_scaler = scaler_min.shape[0]
except FileNotFoundError:
    raise FileNotFoundError("Scaler min/max files not found. "
                            "Please ensure 'train_enemy_lstm.py' has been run to save them.")

# Verify TensorFlow GPU usage (optional, for debugging performance)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# --- Game setup ---
pygame.init()
WIDTH, HEIGHT = 800, 600
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("CATCH ME IF YOU CAN")

# Define Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255) # Bright color for score label
BRIGHT_BLUE = (0, 191, 255) # <--- CHANGED: Bright blue for the score number itself


# --- Custom Font Setup ---
CUSTOM_FONT_PATH = "Debrosee-ALPnL.ttf" # Your font file
FALLBACK_FONT_NAME = "Arial" # Standard system font for fallback and for numbers

FONT_SIZE_TITLE = 60 # For main game title
FONT_SIZE_MEDIUM = 35 # For instructions, game over messages, score label
FONT_SIZE_SMALL = 30 # For player/enemy names

# Font objects for custom font
game_font_title = None
game_font_medium = None
game_font_small = None
# GUARANTEE FIX: Force score_number_font to a reliable system font for numbers
score_number_font = None


try:
    game_font_title = pygame.font.Font(CUSTOM_FONT_PATH, FONT_SIZE_TITLE)
    game_font_medium = pygame.font.Font(CUSTOM_FONT_PATH, FONT_SIZE_MEDIUM)
    game_font_small = pygame.font.Font(CUSTOM_FONT_PATH, FONT_SIZE_SMALL)
    
    # Attempt to load custom font for score numbers, but be prepared to fallback reliably
    # This block is for initial loading, the explicit override below handles the number issue
    score_number_font = pygame.font.Font(CUSTOM_FONT_PATH, FONT_SIZE_MEDIUM) 
    
    print(f"Custom font '{CUSTOM_FONT_PATH}' loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Custom font '{CUSTOM_FONT_PATH}' not found. Falling back to '{FALLBACK_FONT_NAME}' for all text.")
    game_font_title = pygame.font.SysFont(FALLBACK_FONT_NAME, FONT_SIZE_TITLE)
    game_font_medium = pygame.font.SysFont(FALLBACK_FONT_NAME, FONT_SIZE_MEDIUM)
    game_font_small = pygame.font.SysFont(FALLBACK_FONT_NAME, FONT_SIZE_SMALL)
    score_number_font = pygame.font.SysFont(FALLBACK_FONT_NAME, FONT_SIZE_MEDIUM)
except Exception as e:
    print(f"ERROR: Could not load custom font '{CUSTOM_FONT_PATH}'. Falling back to '{FALLBACK_FONT_NAME}' for all text. Error: {e}")
    game_font_title = pygame.font.SysFont(FALLBACK_FONT_NAME, FONT_SIZE_TITLE)
    game_font_medium = pygame.font.SysFont(FALLBACK_FONT_NAME, FONT_SIZE_MEDIUM)
    game_font_small = pygame.font.SysFont(FALLBACK_FONT_NAME, FONT_SIZE_SMALL)
    score_number_font = pygame.font.SysFont(FALLBACK_FONT_NAME, FONT_SIZE_MEDIUM)

# GUARANTEED FIX: Explicitly set the score_number_font to a system font
# This overrides any previous attempt to use the custom font for numbers if it's failing.
print(f"FORCING score number font to '{FALLBACK_FONT_NAME}' to ensure visibility of digits.")
score_number_font = pygame.font.SysFont(FALLBACK_FONT_NAME, FONT_SIZE_MEDIUM)


# Constants for game physics and movement
GRAVITY = 0.5
JUMP_STRENGTH = -10
PLAYER_SPEED = 5
ENEMY_SPEED = 4

# --- Load Game Assets (Background and Character Sprites) ---
BACKGROUND_IMAGE_PATH = "background.png"
scaled_bg_image = None

try:
    original_bg_image = pygame.image.load(BACKGROUND_IMAGE_PATH).convert()
    scaled_bg_image = pygame.transform.scale(original_bg_image, (WIDTH, HEIGHT))
    print(f"Background image '{BACKGROUND_IMAGE_PATH}' loaded successfully.")
except pygame.error as e:
    print(f"ERROR: Could not load background image '{BACKGROUND_IMAGE_PATH}'. Falling back to solid white background. Error: {e}")

PLAYER_SPRITE_PATH = "player_sprite.png"
ENEMY_SPRITE_PATH = "enemy_sprite.png"
DEFAULT_CHAR_WIDTH = 64
DEFAULT_CHAR_HEIGHT = 64
player_sprite_image = None
enemy_sprite_image = None

try:
    player_sprite_image = pygame.image.load(PLAYER_SPRITE_PATH).convert_alpha()
    player_sprite_image = pygame.transform.scale(player_sprite_image, (DEFAULT_CHAR_WIDTH, DEFAULT_CHAR_HEIGHT))
    print(f"Player sprite '{PLAYER_SPRITE_PATH}' loaded successfully.")
except pygame.error as e:
    print(f"ERROR: Could not load player sprite '{PLAYER_SPRITE_PATH}'. Falling back to blue block. Error: {e}")

try:
    enemy_sprite_image = pygame.image.load(ENEMY_SPRITE_PATH).convert_alpha()
    enemy_sprite_image = pygame.transform.scale(enemy_sprite_image, (DEFAULT_CHAR_WIDTH, DEFAULT_CHAR_HEIGHT))
    print(f"Enemy sprite '{ENEMY_SPRITE_PATH}' loaded successfully.")
except pygame.error as e:
    print(f"ERROR: Could not load enemy sprite '{ENEMY_SPRITE_PATH}'. Falling back to red block. Error: {e}")


# --- Player class ---
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        if player_sprite_image:
            self.image = player_sprite_image
        else:
            self.image = pygame.Surface((DEFAULT_CHAR_WIDTH, DEFAULT_CHAR_HEIGHT))
            self.image.fill(BLUE)
        
        self.rect = self.image.get_rect()
        self.rect.center = (WIDTH // 4, HEIGHT - 100)
        self.vel_y = 0
        self.on_ground = True

    def update(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.rect.x -= PLAYER_SPEED
        if keys[pygame.K_RIGHT]:
            self.rect.x += PLAYER_SPEED
        if keys[pygame.K_SPACE] and self.on_ground:
            self.vel_y = JUMP_STRENGTH
            self.on_ground = False

        self.vel_y += GRAVITY
        self.rect.y += self.vel_y

        if self.rect.bottom >= HEIGHT:
            self.rect.bottom = HEIGHT
            self.vel_y = 0
            self.on_ground = True
        
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > WIDTH:
            self.rect.right = WIDTH

# --- Enemy class ---
class Enemy(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        if enemy_sprite_image:
            self.image = enemy_sprite_image
        else:
            self.image = pygame.Surface((DEFAULT_CHAR_WIDTH, DEFAULT_CHAR_HEIGHT))
            self.image.fill(RED)

        self.rect = self.image.get_rect()
        self.rect.center = (WIDTH // 2, HEIGHT - 100)
        self.vel_y = 0
        self.on_ground = True
        self.history = []
        self.sequence_length = 10

        self.prediction_cooldown = 0
        self.prediction_interval = 8
        self.predicted_target_x = self.rect.x

        self.model_input_array = np.zeros((1, self.sequence_length, num_features_scaler), dtype=np.float32)

    def update(self, player_x):
        self.history.append(player_x)
        if len(self.history) > self.sequence_length:
            self.history.pop(0)

        self.prediction_cooldown += 1

        if self.prediction_cooldown >= self.prediction_interval and len(self.history) == self.sequence_length:
            self.prediction_cooldown = 0

            scaled_history = (np.array(self.history).reshape(-1, 1) - scaler_min[0]) / (scaler_max[0] - scaler_min[0])
            self.model_input_array[0, :, 0] = scaled_history.flatten()

            prediction_scaled = model.predict(self.model_input_array, verbose=0)
            self.predicted_target_x = (prediction_scaled[0][0] * (scaler_max[0] - scaler_min[0])) + scaler_min[0]
            
        if self.rect.x < self.predicted_target_x:
            self.rect.x += ENEMY_SPEED
        elif self.rect.x > self.predicted_target_x:
            self.rect.x -= ENEMY_SPEED
        
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > WIDTH:
            self.rect.right = WIDTH

        if self.on_ground:
            if abs(self.rect.x - player_x) > 100 or \
               (player_x < self.rect.x - 50 and self.rect.x > 50) or \
               (player_x > self.rect.x + 50 and self.rect.x < WIDTH - 50):
                self.vel_y = JUMP_STRENGTH
                self.on_ground = False

        self.vel_y += GRAVITY
        self.rect.y += self.vel_y

        if self.rect.bottom >= HEIGHT:
            self.rect.bottom = HEIGHT
            self.vel_y = 0
            self.on_ground = True

score = 0

def draw_text_with_shadow(font_obj, text, text_color, shadow_color, x, y, shadow_offset=(2, 2)):
    shadow_surface = font_obj.render(text, True, shadow_color)
    win.blit(shadow_surface, (x + shadow_offset[0], y + shadow_offset[1]))

    text_surface = font_obj.render(text, True, text_color)
    win.blit(text_surface, (x, y))


def main():
    global score
    run = True
    waiting = True

    clock = pygame.time.Clock() 

    player = Player()
    enemy = Enemy()
    all_sprites = pygame.sprite.Group()
    all_sprites.add(player)
    all_sprites.add(enemy)

    # --- Start Screen Loop ---
    while waiting:
        if scaled_bg_image:
            win.blit(scaled_bg_image, (0, 0))
        else:
            win.fill(WHITE)
        
        title_text_content = "CATCH ME IF YOU CAN"
        start_text_content = "Press 'Y' to Start the Game"
        controls_text_line1 = "CONTROLS: LEFT/RIGHT ARROWS TO MOVE,"
        controls_text_line2 = "SPACEBAR TO JUMP"

        draw_text_with_shadow(game_font_title, title_text_content, WHITE, BLACK, WIDTH // 2 - game_font_title.size(title_text_content)[0] // 2, HEIGHT // 3 - 50)
        draw_text_with_shadow(game_font_medium, start_text_content, WHITE, BLACK, WIDTH // 2 - game_font_medium.size(start_text_content)[0] // 2, HEIGHT // 2)
        
        draw_text_with_shadow(game_font_medium, controls_text_line1, WHITE, BLACK, WIDTH // 2 - game_font_medium.size(controls_text_line1)[0] // 2, HEIGHT // 2 + 50)
        draw_text_with_shadow(game_font_medium, controls_text_line2, WHITE, BLACK, WIDTH // 2 - game_font_medium.size(controls_text_line2)[0] // 2, HEIGHT // 2 + 50 + game_font_medium.get_height() + 5)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_y:
                    waiting = False
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return

    # --- Main Game Loop ---
    score = 0 # Ensure score resets ONLY when game starts (after 'Y' is pressed)
    
    print("--- GAME STARTING ---") 
    print(f"DEBUG: Score initialized to 0.") 

    while run:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        player.update()
        enemy.update(player.rect.x)

        if player.rect.colliderect(enemy.rect):
            run = False

        score += 1 # Score increments every frame
        
        if scaled_bg_image:
            win.blit(scaled_bg_image, (0, 0))
        else:
            win.fill(WHITE)

        all_sprites.draw(win) 
        
        draw_text_with_shadow(game_font_small, "Windy", WHITE, BLACK, player.rect.x + (player.rect.width // 2) - (game_font_small.size("Windy")[0] // 2), player.rect.y - 30)
        draw_text_with_shadow(game_font_small, "Hunter", WHITE, BLACK, enemy.rect.x + (enemy.rect.width // 2) - (game_font_small.size("Hunter")[0] // 2), enemy.rect.y - 30)
        
        display_score = score # Display raw frames for clearer debugging
        
        draw_text_with_shadow(game_font_medium, f"Score: {display_score}", WHITE, BLACK, 10, 10)
        # print(f"DEBUG: Current Live Score: {display_score}") # Removed to de-clutter console if many frames pass

        pygame.display.update()

    # --- Game Over Screen ---
    if scaled_bg_image:
        win.blit(scaled_bg_image, (0, 0))
    else:
        win.fill(WHITE)
    
    game_over_text_content = "Better Luck Next Time. LOSER!"
    
    final_display_score = score 
    final_score_label_text = "FINAL SCORE" # Separated label text
    final_score_number_text = str(final_display_score) # Get score as string

    print(f"--- GAME OVER ---")
    print(f"DEBUG: Final Score (raw frames): {final_display_score}")
    print(f"DEBUG: Final Score string being rendered: '{final_score_number_text}'")

    # Draw the main Game Over message
    draw_text_with_shadow(game_font_medium, game_over_text_content, RED, BLACK, WIDTH // 2 - game_font_medium.size(game_over_text_content)[0] // 2, HEIGHT // 2 - 50)
    
    # Draw "FINAL SCORE" label using custom font
    draw_text_with_shadow(game_font_medium, final_score_label_text, CYAN, BLACK, WIDTH // 2 - game_font_medium.size(final_score_label_text)[0] // 2, HEIGHT // 2 + 10)

    # --- Draw the actual numerical score (FINAL ATTEMPT FOR VISIBILITY - NO BOX) ---
    # Render the actual number string using the FORCED score_number_font (Arial)
    # Changed color to BRIGHT_BLUE
    score_number_surface = score_number_font.render(final_score_number_text, True, BRIGHT_BLUE) # Bright Blue text
    
    # Get its rect and center it below the "FINAL SCORE" label
    score_number_rect = score_number_surface.get_rect(centerx=WIDTH // 2)
    score_number_rect.y = (HEIGHT // 2 + 10) + game_font_medium.get_height() + 5 # 5 pixels below the label

    # Removed the black box drawing:
    # pygame.draw.rect(win, (50, 50, 50), score_number_rect.inflate(20, 10)) 

    # Blit the numerical score directly (no shadow on this element for maximum clarity)
    win.blit(score_number_surface, score_number_rect) 

    pygame.display.update()
    pygame.time.delay(3000)
    pygame.quit()

if __name__ == "__main__":
    main()