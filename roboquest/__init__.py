"""RoboQuest task suite in RoboCasa's own task pattern.

One ``Kitchen`` subclass per task (see ``kitchen.RoboQuestKitchen``), frozen
instances instead of per-reset sampling (see ``registry``), RoboCasa's scenes,
assets and render recipe unchanged. Importing this package pulls in no
simulator; task modules under ``roboquest.tasks`` do.
"""
