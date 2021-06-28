import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models.npmodel import Model
from agents.evagent import Agent
sns.set()

df = pd.read_csv('./data/googyear.csv')

def evalute_model(
    df: pd.DataFrame = df, 
    model_name:object = Model, 
    agent_name:object = Agent, 
    trend_name:str ='Close',
    ):

    trend = df[trend_name].values.tolist()
    window_size = 30
    skip = 1
    initial_money = 10000

    model = model_name(input_size = window_size, layer_size = 500, output_size = 3)
    agent = agent_name(model = model, 
                window_size = window_size,
                trend = trend,
                skip = skip,
                initial_money = initial_money)
    agent.fit(iterations = 500, checkpoint = 10)
    states_buy, states_sell, total_gains, invest = agent.buy()

    fig = plt.figure(figsize = (15,5))
    plt.plot(trend, color='r', lw=2.)
    plt.plot(trend, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
    plt.plot(trend, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
    plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
    plt.legend()
    plt.show()
    return

if __name__ == "__main__":
    evalute_model()