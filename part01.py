#!/usr/bin/env python3

from bs4 import BeautifulSoup
import requests
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import List, Callable, Dict, Any


def distance(a: np.array, b: np.array) -> np.array:
    """
    Function calculates distance between two arrays.

    Arguments:
    a and b -- arrays
    """
    power = np.power(a-b, 2)
    sum = np.sum(power, axis=1)
    return np.sqrt(sum)


def label(x,step):
    """Function helps set labels of x axis """
    N = int(np.round(2*x/np.pi))
    if N == 0:
        return fr"$0$"
    elif N == 2:
        return fr"$\pi$"
    elif N % 2 > 0:
        return fr"$\frac{{{N}}}{{2}}\pi$"
    else:
        return fr"${N // 2}\pi$"

def generate_graph(a: List[float], show_figure: bool = False, save_path: str | None = None):
    """
    Function generates a graph with 3 sinuses

    Arguments:
    a -- array of values
    save_path -- file where the graph is saved
    show_figure -- decides whether to show

    """
    # set needed variables
    x = np.arange(0, 6*np.pi, 0.01)
    a = np.reshape(a, (3, 1))
    fax = np.multiply(np.power(a, 2), np.sin(x))  # matrix 3x189

    # create a figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # set x and y axis
    ax.set_xlim(0, 6*np.pi)
    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi/2))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(label))
    ax.set_xlabel(r"x")
    ax.set_ylabel(r"$f_{a}(x)$")

    # colour space under, and plot
    for i in range(0,3):
        ax.fill_between(x, fax[i], alpha=0.2)
        ax.plot(x, fax[i], label=rf"$y_{a[i][0]}(x)$")

    # set possition of the legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

    if save_path is not None:
        fig.savefig(save_path)

    if show_figure == True:
        plt.show()



def generate_sinus(show_figure: bool = False, save_path: str | None = None):
    """
    Function generating three graphs of sinuses

    Arguments:
    save_path -- file where the graph is saved
    show_figure -- decides whether to show

    """

    # set needed variables
    t = np.arange(0, 100, 0.005) #small enough step in order not disturb visuals
    f1 = 0.5*np.cos(np.pi*t/50)
    f2 = 0.25*(np.sin(np.pi*t) + np.sin(np.pi*t*3/2))

    # create a figure for 3 subplot with shares axis
    fig, axs = plt.subplots(3, 1, sharex=True,sharey=True)

    # set all x and y axis
    axs[0].set_xlim(0, 100)
    axs[0].xaxis.set_major_locator(plt.MultipleLocator(25))
    axs[0].set_ylim(-0.8,0.8)
    axs[0].yaxis.set_major_locator(plt.MultipleLocator(0.4))


    # plot the first two subplots
    axs[0].plot(t, f1)
    axs[0].set_ylabel(rf"$f_{1}(t)$")

    axs[1].plot(t, f2)
    axs[1].set_ylabel(rf"$f_{2}(t)$")

    # last subplot
    f12 = f1 + f2

    # seperate values
    above = np.ma.masked_where(f1 >= f12, f12)
    below = np.ma.masked_where(f1 < f12, f12)

    # divide below in half and plot
    axs[2].plot(t, above, "green", t[t <= 50], below[t <= 50], "red", t[t > 50], below[t > 50], "orange")
    axs[2].set_ylabel(rf"$f_{1}(t) + f_{2}(t)$")

    if save_path is not None:
        fig.savefig(save_path)
    if show_figure == True:
        plt.show()


def download_data() -> Dict[str, List[Any]]:
    """
    Function downloads geographical data of cities 
    
    """

    url = "https://ehw.fit.vutbr.cz/izv/stanice.html"
    rs= requests.get(url)

    # check whether the data were obtained successfully
    if (rs.status_code >= 400):
        print(f'Error: {rs.status_code}')
        exit()

    # using beatifulsoup for easier parsing
    soup = BeautifulSoup(rs.content, "html.parser")

    # need to get adress to get data from the table
    boxes = soup.find_all("div", class_="box")

    for box in boxes:
        script = box.find("script")
        if script:
            script = script.text.split("'")

    new_adress = url.replace("stanice.html",script[1])
    
    rs = requests.get(new_adress)
    if (rs.status_code >= 400):
        print(f'Error in the second request: {rs.status_code}')
        exit()

    
    soup = BeautifulSoup(rs.content, "html.parser")

    # init the dictionary
    data = {'positions': [], 'lats':[],'longs':[],'heights':[]}

    # parsing through soup
    table = soup.find_all("tr")
    for row in table:
        cols = row.find_all("td")
        if len(cols) == 7: # put data into a correct format and store it in dict
            data['positions'].append(cols[0].text.strip())
            data['lats'].append(float(cols[2].text.replace(',', '.').replace('°', '')))
            data['longs'].append(float(cols[4].text.replace(',', '.').replace('°', '')))
            data['heights'].append(float(cols[6].text.replace(',', '.')))

    return(data)


if __name__ == "main":
    generate_graph([7, 4, 3])
    generate_sinus()

