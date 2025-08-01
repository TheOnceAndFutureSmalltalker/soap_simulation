import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.image import AxesImage
import numpy as np
import numpy.typing as npt



def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

class SoapSimulation():
    '''
    Simulation of a bar of soap shrinking under streams of water from a shower head.
    
    Methods
    -------
        constructor(**kwargs)
            Initializes the simulation parameters    
        run_simulation(self, frames: int = 10, interval: int =100, delta_t: float = 1.0)
            Runs the simulation as an animation.
    '''
    def __init__(self, **kwargs):
        '''
        Parameters
        ----------
            soap_bar_length: float 
                in centimeters
            soap_bar_height: float 
                in centimeters
            soap_bar_cell_width: float 
                in centimeters
            soap_bar_density: float 
                milligrams per milliliter (cm^3)
            soap_dissolution_rate:float 
                milligrams of soap dissolved per mil of water per cm^2 of surface area
            soap_ablation_rate: float 
                milligrams of soap ablated per mil of water per second
            number_of_streams: int 
                number of shower head streams affecting the soap bar
            water_flow_rate_per_stream: float 
                milliliters (cm^3) of water per second landing on cm^2 for one stream 
            water_flow_rate_std_dev: float 
                variance in water flow rate above
            stream_droplet_sample_size: int 
                number of droplets of water per stream per iteration
            droplet_dispersion_std_dev: int 
                spread of droplets in centimeters for a stream
            water_splash_rate: float
                fraction of water that splashes off and not available for dissolution
        '''
        
        # soap bar parameters
        self.soap_bar_length: float = clamp(kwargs.get("soap_bar_length", 4.0), 2.0, 10.0)
        self.soap_bar_height: float = clamp(kwargs.get("soap_bar_height", 2.0), 1.0, 4.0)
        self.soap_bar_cell_width: float = clamp(kwargs.get("soap_bar_cell_width", 1.0), 0.1, 1.0)
        self.soap_density: float = clamp(kwargs.get("soap_density", 1000.0), 700.0, 1000.0) 
        self.soap_dissolution_rate: float = clamp(kwargs.get("soap_dissolution_rate", 1.0 ), 0.01, 1.0)
        self.soap_ablation_rate: float = clamp(kwargs.get("soap_ablation_rate", 0.25 ), 0.01, 0.25)

        # water parameters
        self.number_of_streams: int = clamp(kwargs.get("number_of_streams", 5 ), 2, 20)
        self.water_flow_rate_per_stream: float = clamp(kwargs.get("water_flow_rate_per_stream", 1.0), 0.1, 2.0)
        self.water_flow_rate_std_dev = clamp(kwargs.get("water_flow_rate_std_dev", 0.25), 0.0, 0.5)
        self.stream_droplet_sample_size: int = clamp(kwargs.get("stream_droplet_sample_size", 20), 10, 50) 
        self.droplet_dispersion_std_dev: float = clamp(kwargs.get("droplet_dispersion_std_dev", 0.5), 0.1, 1.0)
        self.water_splash_rate: float = clamp(kwargs.get("water_splash_rate", 0.1), 0.05, 0.25)

    def create_soap_bar(self):
        self.soap_mass_per_cell = self.soap_density * pow(self.soap_bar_cell_width, 3)
        cells_in_length: float = int(self.soap_bar_length / self.soap_bar_cell_width)
        cells_in_height: float = int(self.soap_bar_height / self.soap_bar_cell_width)
        self.soap_bar: npt.NDArray = np.ones((cells_in_height, cells_in_length)) * self.soap_mass_per_cell
        # Maintain the remaining mass in milligrams.
        self.mass_remaining: float = np.sum(self.soap_bar) 

    def calculate_water_catch_per_column(self):
        # Determine how much water falls on each cell_width of soap
        
        # First, lets calculate how much water is in each stream over delta_t
        # and considering the thickness of the cross section.
        # Average speed of the water can vary from one iteration to the next
        # according to a gaussian distribution.
        current_water_flow_rate_per_stream: float = random.gauss(self.water_flow_rate_per_stream, self.water_flow_rate_std_dev)
        current_water_flow_rate_per_stream = self.water_flow_rate_per_stream  # remove after testing
        # Calculate water hitting per cm^2 over entire delta_t.
        water_per_stream_over_delta_t: float = current_water_flow_rate_per_stream * self.delta_t 
        # Now scale per stream water by thicknes of the soap slice..
        water_per_stream_over_delta_t_for_thickness = water_per_stream_over_delta_t * self.soap_bar_cell_width
        water_per_droplet: float = water_per_stream_over_delta_t_for_thickness / self.stream_droplet_sample_size

        # Next, evenly distribute the streams over a length that includes 
        # the bar length and half of stream_std_dev on both sides. 
        # This accounts for streams whose center is not directly over the bar, 
        # but some of its droplets will still hit the bar.
        start: float = (self.droplet_dispersion_std_dev * -0.5)
        end: float = self.soap_bar_length + self.droplet_dispersion_std_dev * 0.5
        stream_means: npt.NDArray = np.linspace(start, end, self.number_of_streams)
        
        # Now, find out how many water droplets fall on each cell_width on top of the soap.
        _, cols = self.soap_bar.shape
        droplets_by_column: npt.NDArray = np.zeros((cols,))
        for mean_location in stream_means:
            droplets: npt.NDArray = np.random.normal(mean_location, self.droplet_dispersion_std_dev, self.stream_droplet_sample_size)
            conditional: npt.NDArray = (droplets >= 0) & (droplets < self.soap_bar_length)
            droplets = droplets[conditional]
            droplet_bins = np.arange(self.soap_bar_cell_width, self.soap_bar_length, self.soap_bar_cell_width)
            indices: npt.NDArray  = np.digitize(droplets, droplet_bins)
            droplets_distribution: npt.NDArray = np.bincount(indices, minlength = len(droplet_bins) + 1)
            droplets_by_column += droplets_distribution

        # Finaly, calculate total water hitting each cell_width on the top of the soap.
        self.water_by_column = droplets_by_column * water_per_droplet

    def apply_ablation(self):
        # For the top of each column, start applying the ablation of the water.
        rows, cols = self.soap_bar.shape
        for col in range(0, cols):
            ablated_mass: float = self.water_by_column[col] * self.soap_ablation_rate
            self.mass_remaining -= ablated_mass
            row: int = 0
            # Start reducing each cell_width column of soap by the ablated_mass.
            while ablated_mass and row < rows:
                if ablated_mass <= self.soap_bar[row][col]:
                    self.soap_bar[row][col] -= ablated_mass
                    break
                ablated_mass -= self.soap_bar[row][col]
                self.soap_bar[row][col] = 0.0
                row += 1

    def apply_dissolution(self):
        # For the top and the sides of the soap, calculate how much mass
        # is removed due to dissolution.
        rows, cols = self.soap_bar.shape
        # Calculate how much water is flowing over the top of each column as it runs to both sides.
        # Assume half of the water runs to each side.
        water_flow_by_column: npt.NDArray = self.water_by_column * (1.0 - self.water_splash_rate)
        if cols % 2:
            water_flow_by_column[cols//2-1] += water_flow_by_column[cols//2] * 0.5
            water_flow_by_column[cols//2+1] += water_flow_by_column[cols//2] * 0.5
        for col in range(cols//2-2, -1, -1):
            water_flow_by_column[col] += water_flow_by_column[col+1]
        for col in range(cols//2+1, cols):
            water_flow_by_column[col] += water_flow_by_column[col-1]

        # For each column start applying total mass to be dissolved. 
        # If top cell has less mass than total mass to be dissolved,
        # then apply the remaining to the next lower cell.
        for col in range(0, cols):
            dissolved_mass: float = water_flow_by_column[col] * self.soap_dissolution_rate * pow(self.soap_bar_cell_width, 2)
            self.mass_remaining -= dissolved_mass
            row: int = 0
            while dissolved_mass and row < rows:
                if dissolved_mass <= self.soap_bar[row][col]:
                    self.soap_bar[row][col] -= dissolved_mass
                    break
                dissolved_mass -= self.soap_bar[row][col]
                self.soap_bar[row][col] = 0.0
                row += 1

        # For each row apply mass dissolved on left and right sides.
        # If outer most cell has less mass than total mass dissolved to be dissolved, 
        # then apply the remaining to the next inner most cell
        for row in range(0, rows):
            indices = np.where(self.soap_bar[row] > 0.0)[0]
            if not indices.size:
                continue
            #left side
            col: int = indices[0]
            dissolved_mass: float = water_flow_by_column[col] * self.soap_dissolution_rate * pow(self.soap_bar_cell_width, 2)
            self.mass_remaining -= dissolved_mass
            while dissolved_mass and col < len(indices):
                if dissolved_mass <= self.soap_bar[row][col]:
                    self.soap_bar[row][col] -= dissolved_mass
                    break
                dissolved_mass -= self.soap_bar[row][col]
                self.soap_bar[row][col] = 0.0
                col += 1
            #rigth side
            col = indices[-1]
            dissolved_mass = water_flow_by_column[col] * self.soap_dissolution_rate * pow(self.soap_bar_cell_width, 2)
            self.mass_remaining -= dissolved_mass
            while dissolved_mass and col >= 0:
                if dissolved_mass <= self.soap_bar[row][col]:
                    self.soap_bar[row][col] -= dissolved_mass
                    break
                dissolved_mass -= self.soap_bar[row][col]
                self.soap_bar[row][col] = 0.0
                col -= 1

    def update_soap_bar(self):
        # Figure out how much water is hitting the top of the soap bar
        # for delta_t, and how it is distributed accross the cells, 
        # and then apply ablation and dissolution.
        self.calculate_water_catch_per_column()
        self.apply_ablation()
        self.apply_dissolution()

    def init_animation(self):
        # Hand the initial soap data to the animation.
        self.image.set_data(self.soap_bar)
        return self.image,

    def update_animation(self, frame):
        # Update the soap bar and then update the animation with 
        # the new state of the soap bar.
        self.update_soap_bar()
        self.image.set_data(self.soap_bar)
        # I would be nice to update title of animation with frame number, 
        # mass, and elapsed time, but that requires not Blitting, which is slow.
        # So, just log the values here.
        elapsed_time: float = self.delta_t * frame
        mass_remaining_grams: float = self.mass_remaining / 1000.0
        one_based_frame: int = frame + 1
        print(f"frame: {one_based_frame}, elapsed time: {elapsed_time} sec, mass remaining: {mass_remaining_grams:.3f} g")
        return self.image,

    def run_simulation(self, frames: int = 10, interval: int =100, delta_t: float = 1.0) -> None:
        '''
        Runs the simulation as an animation.  Background is in white and the soap bar is int black.
        A mass is removed from a cell of the bar of soap, the color becomes lighter.  When all of
        the mass has been removed from a cell, it is white, matching the background.

        Parameters
        ----------
            frames: int
                the number of iterations for the simulation
            interval: int
                the millisecond delay between frames
            delta_t: float
                the amount of time represented by each frame
        '''
        self.delta_t = delta_t
        # Always recreate a new soap bar so that the simulation can be run 
        # multiple times on a single instance.
        self.create_soap_bar()
        fig, ax = plt.subplots()
        fig.suptitle("Soap Simulation")
        self.image: AxesImage = ax.matshow(self.soap_bar, cmap='gray_r', vmin=0.0, vmax=self.soap_mass_per_cell)
        # Reference to animation is required otherwise it might 
        # be garbage collected mid simulation!
        animation = FuncAnimation(fig, self.update_animation, frames=frames, init_func=self.init_animation, interval=interval, blit=True, repeat=False)
        plt.colorbar(self.image)
        plt.show()
        print("Simulation complete.")
        


if __name__ == "__main__":
    ss = SoapSimulation(
        # soap bar parameters
        soap_bar_length = 8.0,       # centimeters, cm
        soap_bar_height = 2.0,       # centimeters, cm
        soap_bar_cell_width = 0.25,  # centimeters, cm
        soap_density = 1000.0,       # miligrams per cm^3
        soap_dissolution_rate = 1.0, # mg removed per cm^3 of water flowing over cm^2 of surface
        soap_ablation_rate = 0.25,   # mg removed per cm^3 water per second

        # water dynamics parameters
        number_of_streams = 5,
        water_flow_rate_per_stream = 1.0,   # mil of water hitting cm^2 area per second
        water_flow_rate_std_dev = 1.0,      # spread of the flow rate
        stream_droplet_sample_size = 20,    # number of droplest in stream per iteration
        droplet_dispersion_std_dev = 0.5,   # spread of droplets from mean of stream
        water_splash_rate = 0.1             # fraction of water splashing off 
    ) 
    # This runs for fifteen minutes in sim time, about 20 seconds real time.
    ss.run_simulation(frames=180, interval=100, delta_t=5.0)


    
    

    


