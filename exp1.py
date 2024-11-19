import numpy as np 
import matplotlib.pyplot as plt 

def plot_sine_wave():     
    x = np.linspace(0, 2*np.pi, 100)     
    y = np.sin(x)           
    plt.figure(figsize=(10, 6))     
    plt.plot(x, y, 'b-', label='sin(x)')     
    plt.title('Simple Sine Wave')     
    plt.xlabel('x')     
    plt.ylabel('sin(x)')     
    plt.grid(True)     
    plt.legend()    
    plt.show() 
    
if __name__ == "__main__": 
    print("Python environment is working correctly!")     
    print(f"NumPy version: {np.__version__}")     
    plot_sine_wave() 
