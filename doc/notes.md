# Notes

## Misc
- Paddles have different colors
- Would it make sense to have the AI player always on the same side? This could
be achieved by applying an image rotation and swapping paddle colors e.g. for
the right player <-> The environment actually already does this.

## Environment
- 3 actions
  - up
  - down
  - stay in place

- State: 3 channel image (RBG), the wimblepong env already does some pre-processing:
  1. Removes the scoreboard from the top corner -> 200x200 image (README.md has incorrect image dimensions)
  2. Rotates image for the right player
  3. Inverts paddle colors for the right player -> the game always looks like the agent would be playing as the left player

- Players
  - player 1 == left
  - player 2 == right


## Approach

### Pre-processing
1. Convert to 1 channel: PIL img.convert('L')
2. Frame stacking / difference frames


### ML
- DQN-CNN
- ACER-CNN
