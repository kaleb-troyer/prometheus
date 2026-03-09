import matplotlib.pyplot as plt
import scipy.interpolate
import numpy as np
from scipy.stats import norm
from scipy.integrate import cumulative_trapezoid, trapezoid

def make_flux_limits(limit_list, width):
    # get the number of sections
    n_sections = len(limit_list)-1
    
    # make the points
    sec_pts = np.linspace( -width/2, width/2,n_sections+1 )
    
    limit_fun = scipy.interpolate.interp1d(sec_pts, limit_list,'previous')

    return limit_fun

def make_2D_images_power_basis(n_images,power_min=0.001,power_max=2,sigma_min=0.025,sigma_max=0.25):
    """
    Makes a dictionary of 2D gaussian images of size n_images with sigmas evenly distributed. 
    All images start centered at zero (mean = zero).
    n_images - (int) number of desired images
    power_max - (kW) the maximum possible power 
    power_min - (kW) the minimum possible power 
    sigma_min- (m) the minimum image standard deviation
    sigma_max- (m) the maximum image standard deviation
    """
    
    sigmas=np.linspace(sigma_min,sigma_max,n_images)
    powers=np.linspace(power_min,power_max,n_images)
    image_dict={}
    name=0
    for sigma,power in zip(sigmas,powers):
        # image_info=[]
        # Define mean and standard deviation
        mu = 0
        # Create a range of x values
        x = np.linspace(mu - 2*sigma, mu + 2*sigma, 100)
        # Calculate the PDF
        pdf = norm.pdf(x, mu, sigma)
        cumulative_int= cumulative_trapezoid(pdf, x)
        P_frac_per_section   = np.diff(np.concatenate( (np.array([0]),cumulative_int) ) )
        P_per_section        = P_frac_per_section*power
        fluxes_per_section     = P_per_section/np.diff(x)
        # pdf = pdf/pdf.max()
        image_dict[name]=[mu,x[1:],fluxes_per_section]
        name=name+1
    return image_dict

def make_2D_images_peak_flux_basis(n_images,flux_min=0.001,flux_max=2,sigma_min=0.025,sigma_max=0.25):
    """
    Makes a dictionary of 2D gaussian images of size n_images with sigmas evenly distributed. 
    All images start centered at zero (mean = zero).
    n_images - (int) number of desired images
    power_max - (kW) the maximum possible power 
    power_min - (kW) the minimum possible power 
    sigma_min- (m) the minimum image standard deviation
    sigma_max- (m) the maximum image standard deviation
    """
    
    sigmas=np.linspace(sigma_min,sigma_max,n_images)
    fluxes=np.linspace(flux_min,flux_max,n_images)
    image_dict={}
    name=0
    for sigma,flux in zip(sigmas,fluxes):
        # image_info=[]
        # Define mean and standard deviation
        mu = 0
        # Create a range of x values
        x = np.linspace(mu - 2*sigma, mu + 2*sigma, 1000)
        # Calculate the PDF
        pdf = norm.pdf(x, mu, sigma)
        pdf_normd = pdf/pdf.max()
        flux_prof = pdf_normd*flux
        image_dict[name]=[mu,x,flux_prof]
        name=name+1
    return image_dict

def make_2D_flux_fun(ctr,flux,sigma):
    """
    Makes a 2d interpolation that returns the approximate flux distribution at any queried point. Returns values for anything within 2 standard deviations of the center
    ctr - (m) coordinate of the image's center
    flux - (kW/m2) peak flux of the image 
    sigma - (m) the image standard deviation
    """
    mu = ctr
    # Create a range of x values
    x = np.linspace(mu - 2*sigma, mu + 2*sigma, 1000)
    # Calculate the PDF
    pdf = norm.pdf(x, mu, sigma)
    pdf_normd = pdf/pdf.max()
    flux_prof = pdf_normd*flux
    flux_fun = scipy.interpolate.interp1d(x,flux_prof,fill_value=0,bounds_error=False)    
    return flux_fun

def check_fluxes(images,points):
    """
    takes a number of image objects and sums together their flux contributions at the desired points
    images - (list) of image objects
    points - (array of coordinates) the desired x coordinates to sum at
    ---
    returns: an array of cumulative fluxes (kW/m2)
    """
    total_fluxes=np.array([])
    for point in points:
        total_flux=0
        for image in images:
            total_flux=total_flux+image.flux_fun(point)
        total_fluxes=np.concatenate((total_fluxes,np.array([total_flux])))
    return total_fluxes

def calculate_2D_power(fluxes,pts):
    """
    calculates 2D power assuming 1m depth into page.
    fluxes - (kW/m^2) array of fluxes at points
    pts    - (m) coordinates of each flux evaluation point
    """
    power=trapezoid(fluxes,pts)
    return power

def calculate_heuristic_performance(fluxes_ctr, fluxes_heur,bounds, pts):
    """
    calculates the heuristic's aiming efficiency in terms of incident power change while accounting for bound violations
    fluxes_ctr - (kW/m^2) the flux profile when each image is aimed at the center
    fluxes_heur- (kW/m^2) the flux profile resulting from the heuristic
    bounds     - (kW/m^2) the flux boundaries along pts
    pts        - (m)      the coordinates of each flux evaluation point
    ---
    aiming_eta - (-)    efficiency
    slack_power- (kW)   total integrated difference between actual flux and bounds
    """
    power_at_ctr            =calculate_2D_power(fluxes_ctr,pts)
    power_after_heuristic   =calculate_2D_power(fluxes_heur,pts)
    aiming_eta              =power_after_heuristic/power_at_ctr

    slack_flux  =fluxes_heur - bounds(pts) # assuming that fluxes_heur has already been evaluated with spacing=pts
    over_flux   =np.clip(slack_flux, a_min=0, a_max=1e6)  # only evaluate positive fluxes
    under_flux  =np.clip(slack_flux, a_min=-1e6, a_max=0) # only evaluate negative fluxes

    over_power  =calculate_2D_power(over_flux,pts)
    under_power =calculate_2D_power(under_flux,pts)

    return aiming_eta, over_power, under_power

## helper definition borrowed from HALOS
def move_2d_array_from_ctr(data, x, y, constant=False):
            """
            Shifts the array in two dimensions while setting rolled values to constant
    
            Parameters
            ----------
            data : 2D Array
                The 2d numpy array to be shifted
            x : int
                The new desired x location
            y : int
                TThe new desired y location
            constant :  optional
                The constant to replace rolled values with
    
            Returns
            -------
            shifted_array : 2D Array
                The shifted array with "constant" where roll occurs
    
            """
            nx  =data.shape[0]
            ny  =data.shape[1]
            ctr_x = int(nx/2)-1
            ctr_y = int(ny/2)-1

            dx  =x-ctr_x
            dy  =y-ctr_y

            shifted_data = np.roll(data, dx, axis=1)
            if dx < 0:
                shifted_data[:, dx:] = constant
            elif dx > 0:
                shifted_data[:, 0:dx] = constant
        
            shifted_data = np.roll(shifted_data, dy, axis=0)
            if dy < 0:
                shifted_data[dy:, :] = constant
            elif dy > 0:
                shifted_data[0:dy, :] = constant
            return shifted_data



if __name__ == '__main__':
    ## test bound making function
    width=11
    epsilon = 0.01
    limit_list=(100,300,500,700,1100,1100,700,500,300,100,100) # last one needs to include right and left bounds 
    n_sections = len(limit_list)
    limit_fun = make_flux_limits(limit_list, width)
    Xs = np.linspace( -width/2+epsilon, width/2-epsilon,100000 ) 
    Ys = limit_fun(Xs)
    plt.plot(Xs, Ys)
    plt.show()

    ## test image making function
    image_dict=make_2D_images_power_basis(n_images=5,power_min=2,power_max=2,sigma_min=0.025,sigma_max=0.25)
    fig,ax = plt.subplots()
    for key in image_dict.keys():
        ax.plot(image_dict[key][1],image_dict[key][2],label=f"image {key}")
    ax.set_xlabel('position [m]')
    ax.set_ylabel('flux (kW/m$^2$)')
    ax.legend()
    # fig.savefig('imgs/image_maker_power_range.png')
    plt.show()

    ## test image making function with peak flux basis
    image_dict=make_2D_images_peak_flux_basis(n_images=5,flux_min=2,flux_max=2,sigma_min=0.025,sigma_max=0.25)
    fig2,ax2 = plt.subplots()
    for key in image_dict.keys():
        ax2.plot(image_dict[key][1],image_dict[key][2],label=f"image {key}")
    ax2.set_xlabel('position [m]')
    ax2.set_ylabel('flux (kW/m$^2$)')
    ax2.legend()
    # fig2.savefig('imgs/image_maker_flux_range.png')
    plt.show()

    ## test image function with peak flux basis
    image_fun=make_2D_flux_fun(ctr=0,flux=2,sigma=0.025)
    ax2.plot(Xs,image_fun(Xs))
    ax2.set_xlabel('position [m]')
    ax2.set_ylabel('flux (kW/m$^2$)')
    ax2.set_xlim(-0.5,0.5)
    # fig2.savefig('imgs/image_fun_maker_flux_range.png')
    plt.show()

