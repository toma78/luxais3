# luxais3
Lux AI season 3 Kaggle competition



This a bot where I tried a "task oriented" approach where at each turn, each agent will be assigned a task (e.g. explore, sap, collect points, ...). It is a rule based bot and the plan to add some sort of RL later never realized in the end. Still, I think this systematic approach is flexible enough to adjust it to almost any strategy.

Bot is divided in several classes:

    agent.py is main class and also container for all other objects
    environment.py is tracking the map trough game and infering most of random parameters
    strategy.py is the class containing most of heuristics used to adjust bot behaviour
    tactics.py contains some inference, opponent tracking and sapping heuristics
    task_manager.py is searching for viable tasks and assigns them to agent based on priority
    unit_manager.py some inference and unit tracking
    utils.py precalculated utility data structures and functions

In the end, not all game mechanic was inferred, strategy could probably be much more refined and entire code could use some refactoring. Task abstraction was very flexible so change in strategy was mostly a matter of finding the right numbers in priority calculation. One of the main limitations of this approach is that coordination of units can raise up to quadratic complexity in number of units.

