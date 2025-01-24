# Blackjack game

## Description
The game is played with an infinite deck (or with replacement). The game starts with the dealer have one face up and one face down card, while the player have two face up cards. The player can request additional card (hit, `action=1`) until they decide to stop (stick, `action=0`) or exceed 21 (bust, immediate loss). 

- Face cards (King, Queen, Jack) have the point value of 10
- Aces can either count as 11 (usable ace) or 1
- Numerical cards (2-9) has value equal to their number

## Objective
Beat the dealer by obtaining cards that sums to closer to 21 (without going over it) than the dealer's card.

## Action Space
Stick (0) and hit (1).

## Observation Space
3-tuples containing player's current sum, value of dealer's one showing card (1-10 where 1 is ace), and whether the player holds a usable ace (0 or 1).

## Rewards
+1 for win, -1 for lose, 0 for draw.