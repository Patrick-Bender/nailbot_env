from gym.envs.registration import register

register(id='nailbot-v0',
entry_point='nail_bot.envs:NailbotEnv')
