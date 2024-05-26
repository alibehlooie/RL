#del model # remove to demonstrate saving and loading

#model = SAC.load("sac_pendulum")

#obs, info = env.reset()
#while True:
    #action, _states = model.predict(obs, deterministic=True)
    #obs, reward, terminated, truncated, info = env.step(action)
    ##if terminated or truncated:
        #obs, info = env.reset()