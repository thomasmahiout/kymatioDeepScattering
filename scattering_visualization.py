import torch
import os
import scipy.io.wavfile
from scipy.io.wavfile import read
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from kymatio import Scattering1D
from kymatio.datasets import fetch_fsdd
from kymatio.scattering1d.filter_bank import calibrate_scattering_filters, scattering_filter_factory

from function.signal_visualization import *

def plot_multi_order_scattering(x, J, Q, order1_frequency_axis=[],
                                normalise_1=False, normalise_2=False, epsilon_order_1 = 1*10**-6, epsilon_order_2 = 1*10**-6,
                                frequency_normalisation_order_1_vector=None, frequency_normalisation_order_2_vector=None):

    x = torch.from_numpy(x).float()
    x /= x.abs().max()
    x = x.view(1, -1)

    T = x.shape[-1]
    scattering = Scattering1D(J, T, Q=Q, average=True, oversampling=0, vectorize=True)
    Sx = scattering.forward(x)
    Sx_abs = scattering.forward(np.abs(x))
    meta = Scattering1D.compute_meta_scattering(J, Q)
    order0 = (meta['order'] == 0)
    order1 = (meta['order'] == 1)
    order2 = (meta['order'] == 2)

    fig = make_subplots(
        rows=3, cols=6, column_widths=[0.4,0.4,0.4,0.4,0.4,0.4], row_heights=[0.2,0.2,0.2],
        specs = [[{"type": "Scatter"}, {"type": "Heatmap"}, {"type": "Heatmap"},{"type": "Heatmap"}, {"type": "Heatmap"}, {"type": "Heatmap"}],
        [{"type": "Scatter"}, {"type": "Scatter"}, {"type": "Scatter"}, {"type": "Scatter"}, {"type": "Scatter"}, {"type": "Scatter"}],
        [None, {"type": "Scatter"}, {"type": "Scatter"}, {"type": "Scatter"}, {"type": "Scatter"}, {"type": "Scatter"}]],
        subplot_titles = (
        'Temporal signal', 'Scattering Order 1', 'Scattering Order 1 Normalised', 'Scattering Order 2', 'Scattering Order 2 Normalised', 'Order 2 Frequency',
        "Scattering Order 0", 'Scattering Order 1 mean', 'Scattering Order 1 mean Normalised', 'Scattering Order 2 mean', 'Scattering Order 2 mean Normalised', 'Order 2 Frequency mean',
        None, 'Scattering Order 1 max', 'Scattering Order 1 max Normalised', 'Scattering Order 2 max', 'Scattering Order 2 max Normalised', 'Order 2 Frequency max'))

    fig.add_trace(
        go.Scatter(y=x[0,:].numpy(), name="Negative"),
        row=1, col=1)
    fig.add_trace(
        go.Scatter(y=Sx[0,order0,:].numpy().ravel(), name="Negative"),
        row=2, col=1)

    if normalise_1:
        Sx1 = normalise_order1(Sx, Sx_abs, order0, order1, order2, epsilon_order_1, frequency_normalisation_order_1_vector=frequency_normalisation_order_1_vector)
    else:
        Sx1 = Sx[0,order1,:]

    if (len(order1_frequency_axis)!=0):
        fig.add_trace(
            go.Heatmap(z=scale_value(Sx[0,order1,:].numpy()), y=order1_frequency_axis, colorscale='Viridis', showscale=False),
            row=1, col=2)
        fig.add_trace(
            go.Heatmap(z=scale_value(Sx1.numpy()), y=order1_frequency_axis, colorscale='Viridis', showscale=False),
            row=1, col=3)
    else:
        fig.add_trace(
            go.Heatmap(z=scale_value(Sx[0,order1,:].numpy()), colorscale='Viridis', showscale=False),
            row=1, col=2)
        fig.update_yaxes(autorange="reversed", row=1, col=2)
        fig.add_trace(
            go.Heatmap(z=scale_value(Sx1.numpy()), colorscale='Viridis', showscale=False),
            row=1, col=3)
        fig.update_yaxes(autorange="reversed", row=1, col=3)
    fig.add_trace(
        go.Scatter(y=np.mean(scale_value(Sx[0,order1,:].numpy()), axis=1), name="Negative"),
        row=2, col=2)
    fig.add_trace(
        go.Scatter(y=np.max(scale_value(Sx[0,order1,:].numpy()), axis=1), name="Negative"),
        row=3, col=2)
    fig.add_trace(
        go.Scatter(y=np.mean(scale_value(Sx1.numpy()), axis=1), name="Negative"),
        row=2, col=3)
    fig.add_trace(
        go.Scatter(y=np.max(scale_value(Sx1.numpy()), axis=1), name="Negative"),
        row=3, col=3)

    if normalise_2:
        Sx2 = normalise_order2(J, Q, Sx, order0, order1, order2, epsilon_order_2, frequency_normalisation_order_2_vector=frequency_normalisation_order_2_vector)
    else:
        Sx2 = Sx[0,order2,:]

    fig.add_trace(
        go.Heatmap(z=scale_value(Sx[0,order2,:].numpy()), colorscale='Viridis', showscale=False),
        row=1, col=4)
    fig.update_yaxes(autorange="reversed", row=1, col=4)
    fig.add_trace(
        go.Heatmap(z=scale_value(Sx2.numpy()), colorscale='Viridis', showscale=False),
        row=1, col=5)
    fig.update_yaxes(autorange="reversed", row=1, col=5)
    fig.add_trace(
        go.Scatter(y=np.mean(scale_value(Sx[0,order2,:].numpy()), axis=1), name="Negative"),
        row=2, col=4)
    fig.add_trace(
        go.Scatter(y=np.max(scale_value(Sx[0,order2,:].numpy()), axis=1), name="Negative"),
        row=3, col=4)
    fig.add_trace(
        go.Scatter(y=np.mean(scale_value(Sx2.numpy()), axis=1), name="Negative"),
        row=2, col=5)
    fig.add_trace(
        go.Scatter(y=np.max(scale_value(Sx2.numpy()), axis=1), name="Negative"),
        row=3, col=5)
    fig.update_layout(showlegend=False)

    Sx2_Bis = select_frequency(Sx2, T, J, Q, index_frequency=None)

    fig.add_trace(
        go.Heatmap(z=scale_value(Sx2_Bis), colorscale='Viridis', showscale=False),
        row=1, col=6)
    fig.update_yaxes(autorange="reversed", row=1, col=6)
    fig.add_trace(
        go.Scatter(y=np.mean(scale_value(Sx2_Bis), axis=1), name="Negative"),
        row=2, col=6)
    fig.add_trace(
        go.Scatter(y=np.max(scale_value(Sx2_Bis), axis=1), name="Negative"),
        row=3, col=6)
    fig.show()

def scale_value(X):
    return np.log10(X + 1)

def normalise_order1(Sx, Sx_abs, order0, order1, order2, epsilon_order_1, frequency_normalisation_order_1_vector=None):
    Sx1 = Sx[0,order1,:]
    if (len(frequency_normalisation_order_1_vector) != 0):
        Sx1 = Sx1*torch.from_numpy(1/(frequency_normalisation_order_1_vector[:, None])).float()
    Sx1 = Sx1/(Sx_abs[0,order0,:] + epsilon_order_1)
    """
    if (len(frequency_normalisation_order_1_vector) != 0):
        Sx1 = Sx1/(Sx_abs[0,order0,:]*torch.from_numpy(frequency_normalisation_order_1_vector[:, None]).float() + epsilon_order_1)
    else:
        Sx1 = Sx1/(Sx_abs[0,order0,:] + epsilon_order_1)
    """
    return Sx1

def normalise_order2(J, Q, Sx, order0, order1, order2, epsilon_order_2, frequency_normalisation_order_2_vector=None):
    sigma_low, xi1s, sigma1s, j1s, xi2s, sigma2s, j2s = calibrate_scattering_filters(J, Q)
    index_scattering_order_1 = []
    for n1 in range(len(xi1s)):
        for n2 in range(len(xi2s)):
            if j2s[n2] > j1s[n1]:
                index_scattering_order_1.append(n1)
    Sx2 = Sx[0,order2,:]
    if (len(frequency_normalisation_order_2_vector) != 0):
        Sx2 = Sx2*torch.from_numpy(1/(frequency_normalisation_order_2_vector[:, None])).float()
    for i in range(len(index_scattering_order_1)):
        Sx2[i] = Sx2[i]/(Sx[0,order1,:][index_scattering_order_1[i]] + epsilon_order_2)
    """
    if (len(frequency_normalisation_order_2_vector) != 0):
        for i in range(len(index_scattering_order_1)):
            Sx2[i] = Sx2[i]/(Sx[0,order1,:][index_scattering_order_1[i]]*torch.from_numpy(frequency_normalisation_order_2_vector[i, None]).float() + epsilon_order_2)
    else:
        for i in range(len(index_scattering_order_1)):
            Sx2[i] = Sx2[i]/(Sx[0,order1,:][index_scattering_order_1[i]] + epsilon_order_2)
    """
    return Sx2

def select_frequency(Sx2, T, J, Q, index_frequency=None):
    sigma_low, xi1s, sigma1s, j1s, xi2s, sigma2s, j2s = calibrate_scattering_filters(J, Q)
    index_scattering_order_1 = []
    index_scattering_order_2 = []
    for n1 in range(len(xi1s)):
        for n2 in range(len(xi2s)):
            if j2s[n2] > j1s[n1]:
                index_scattering_order_1.append(n1)
                index_scattering_order_2.append(n2)

    Sx2_Bis = []
    if index_frequency!= None:
        for i in range(len(index_scattering_order_1)):
            if index_scattering_order_1[i] == index_frequency:
                Sx2_Bis.append(Sx2[i])
    else:
        Sx2_container = [ [] for _ in range(len(xi2s))]
        for i in range(len(index_scattering_order_1)):
            Sx2_container[index_scattering_order_2[i]].append(Sx2[i].numpy())
        for i in range(len(xi2s)):
            if len(Sx2_container[i])>0:
                Sx2_Bis.append(np.mean(Sx2_container[i], axis=0))
    return np.asarray(Sx2_Bis)

def normalised_frquency_vector_proba(T, J, Q):
    phi_f, psi1_f, psi2_f, _ = scattering_filter_factory(int(np.log2(T)), J, Q)
    frequency_normalisation_order_1_vector = []
    for psi_f in psi1_f:
        norm = np.sqrt(sum(psi_f[0]**2))/(2**psi_f['j']) #sum(psi_f[0])**(3/2)
        frequency_normalisation_order_1_vector.append(norm)
    #fig = px.scatter(y=frequency_normalisation_order_1_vector)
    #fig.show()

    sigma_low, xi1s, sigma1s, j1s, xi2s, sigma2s, j2s = calibrate_scattering_filters(J, Q)
    index_scattering_order_1 = []
    index_scattering_order_2 = []
    for n1 in range(len(xi1s)):
        for n2 in range(len(xi2s)):
            if j2s[n2] > j1s[n1]:
                index_scattering_order_1.append(n1)
                index_scattering_order_2.append(n2)

    frequency_normalisation_order_2_vector = []
    for i in range(len(index_scattering_order_1)):
        psi2 = psi2_f[index_scattering_order_2[i]]
        psi1 = psi1_f[index_scattering_order_1[i]]
        print("index " + str(i) + " psi1 : " + str(psi1['j']) + " psi2 : " + str(psi2['j']))
        norm = np.sqrt(sum(psi1[0]**2))*np.sqrt(sum(psi2[0]**2))/(2**psi2['j'])
        #norm = np.sqrt(sum(psi2[0]**2))/(2**psi2['j'])
        frequency_normalisation_order_2_vector.append(norm)

    return np.asarray(frequency_normalisation_order_1_vector), np.asarray(frequency_normalisation_order_2_vector)

def normalised_frquency_vector(x, J, Q, epsilon_order_1 = 1*10**-6, epsilon_order_2 = 1*10**-6):
    x = torch.from_numpy(x).float()
    x /= x.abs().max()
    x = x.view(1, -1)
    T = x.shape[-1]
    scattering = Scattering1D(J, T, Q=Q, average=True, oversampling=0, vectorize=True)
    Sx = scattering.forward(x)
    Sx_abs = scattering.forward(np.abs(x))
    meta = Scattering1D.compute_meta_scattering(J, Q)
    order0 = (meta['order'] == 0)
    order1 = (meta['order'] == 1)
    order2 = (meta['order'] == 2)

    Sx1 = normalise_order1(Sx, Sx_abs, order0, order1, order2, epsilon_order_1, frequency_normalisation_order_1_vector=[])
    Sx2 = normalise_order2(J, Q, Sx, order0, order1, order2, epsilon_order_2, frequency_normalisation_order_2_vector=[])
    return np.mean(scale_value(Sx1.numpy()), axis=1), np.mean(scale_value(Sx2.numpy()), axis=1)

def get_order1_frequency(T, J, Q, fs):
    phi_f, psi1_f, psi2_f, _ = scattering_filter_factory(int(np.log2(T)), J, Q)
    order1_frequency_axis = []
    for psi_f in psi1_f:
        order1_frequency_axis.append(psi_f['xi']*fs)
    return order1_frequency_axis

# First, the number of samples, `T`, is given by the size of our input `x`.
# The averaging scale is specified as a power of two, `2**J`. Here, we set
# `J = 6` to get an averaging, or maximum, scattering scale of `2**6 = 64`
# samples. Finally, we set the number of wavelets per octave, `Q`, to `16`.
# This lets us resolve frequencies at a resolution of `1/16` octaves.

def main():
    base_dir = '../SimulatorOfUnderwaterSound/data'
    model_path = base_dir + "/white_noise_propeller/binary_2205_With_Delay_Single_Signature_White_Noise_0.4/train/compacted/"
    model_path2 = base_dir + "/standard/test/train/compacted/"

    normalise_1 = True
    normalise_2 = True
    J = 9
    Q = 6
    epsilon_order_1 = 1*10**-12
    epsilon_order_2 = 1*10**-2

    fs = 2205
    dt = 5.0
    T = int(fs*dt)
    t = np.linspace(0, 5, T)
    order1_frequency_axis = get_order1_frequency(T, J, Q, fs)

    signal_pure = 0.1*np.cos(2*np.pi*t*250)
    modulation = (1.2 + 1*np.cos(2*np.pi*t*1.8 + 1))
    modulation2 = (1.2 + 1*np.cos(2*np.pi*t*28.8 + 1))
    random = norm.ppf(np.random.rand(1, T))[0]
    modulated_pure_signal = modulation*signal_pure
    cavitation = modulation*random
    cavitation2 = modulation2*random
    frequency_normalisation_order_1_vector, frequency_normalisation_order_2_vector = normalised_frquency_vector_proba(T, J, Q)

    file_path = model_path + "simulationType_0-0-0_6.wav"
    _, x = read(file_path)

    plot_multi_order_scattering(x, J, Q, order1_frequency_axis=order1_frequency_axis, normalise_1=normalise_1, normalise_2=normalise_2,
    epsilon_order_1=epsilon_order_1, epsilon_order_2=epsilon_order_2, frequency_normalisation_order_1_vector=frequency_normalisation_order_1_vector, frequency_normalisation_order_2_vector=frequency_normalisation_order_2_vector)

    xs = x.copy() + signal_pure
    plot_multi_order_scattering(xs, J, Q, order1_frequency_axis=order1_frequency_axis, normalise_1=normalise_1, normalise_2=normalise_2,
        epsilon_order_1=epsilon_order_1, epsilon_order_2=epsilon_order_2, frequency_normalisation_order_1_vector=frequency_normalisation_order_1_vector, frequency_normalisation_order_2_vector=frequency_normalisation_order_2_vector)

    xs = signal_pure
    plot_multi_order_scattering(xs, J, Q, order1_frequency_axis=order1_frequency_axis, normalise_1=normalise_1, normalise_2=normalise_2,
    epsilon_order_1=epsilon_order_1, epsilon_order_2=epsilon_order_2, frequency_normalisation_order_1_vector=frequency_normalisation_order_1_vector, frequency_normalisation_order_2_vector=frequency_normalisation_order_2_vector)

    xm = x.copy() + modulated_pure_signal
    plot_multi_order_scattering(xm, J, Q, order1_frequency_axis=order1_frequency_axis, normalise_1=normalise_1, normalise_2=normalise_2,
    epsilon_order_1=epsilon_order_1, epsilon_order_2=epsilon_order_2, frequency_normalisation_order_1_vector=frequency_normalisation_order_1_vector, frequency_normalisation_order_2_vector=frequency_normalisation_order_2_vector)

    xm = modulated_pure_signal
    plot_multi_order_scattering(xm, J, Q, order1_frequency_axis=order1_frequency_axis, normalise_1=normalise_1, normalise_2=normalise_2,
    epsilon_order_1=epsilon_order_1, epsilon_order_2=epsilon_order_2, frequency_normalisation_order_1_vector=frequency_normalisation_order_1_vector, frequency_normalisation_order_2_vector=frequency_normalisation_order_2_vector)

    xc = cavitation
    plot_multi_order_scattering(xc, J, Q, order1_frequency_axis=order1_frequency_axis, normalise_1=normalise_1, normalise_2=normalise_2,
    epsilon_order_1=epsilon_order_1, epsilon_order_2=epsilon_order_2, frequency_normalisation_order_1_vector=frequency_normalisation_order_1_vector, frequency_normalisation_order_2_vector=frequency_normalisation_order_2_vector)

    xc = cavitation2
    plot_multi_order_scattering(xc, J, Q, order1_frequency_axis=order1_frequency_axis, normalise_1=normalise_1, normalise_2=normalise_2,
    epsilon_order_1=epsilon_order_1, epsilon_order_2=epsilon_order_2, frequency_normalisation_order_1_vector=frequency_normalisation_order_1_vector, frequency_normalisation_order_2_vector=frequency_normalisation_order_2_vector)

    """
    file_path = model_path + "simulationType_4-100-1_255.wav"
    _, x = read(file_path)
    plot_multi_order_scattering(x, J, Q, order1_frequency_axis=order1_frequency_axis, normalise_1=normalise_1, normalise_2=normalise_2,
    epsilon_order_1=epsilon_order_1, epsilon_order_2=epsilon_order_2, frequency_normalisation_order_1_vector=frequency_normalisation_order_1_vector, frequency_normalisation_order_2_vector=frequency_normalisation_order_2_vector)

    file_path = model_path2 + "simulationType_0_4.wav"
    _, x = read(file_path)
    plot_multi_order_scattering(x, J, Q, normalise_1=normalise_1, normalise_2=normalise_2)
    #signal_visualization(x, "Ambiant Noise", fs=2205)

    file_path = model_path2 + "simulationType_1_4.wav"
    _, x = read(file_path)
    plot_multi_order_scattering(x, J, Q, normalise_1=normalise_1, normalise_2=normalise_2)
    #signal_visualization(x, "Merchant Ship 1", fs=2205)

    file_path = model_path2 + "simulationType_2_4.wav"
    _, x = read(file_path)
    plot_multi_order_scattering(x, J, Q, normalise_1=normalise_1, normalise_2=normalise_2)
    #signal_visualization(x, "Merchant Ship 2", fs=2205)

    file_path = model_path2 + "simulationType_3_4.wav"
    _, x = read(file_path)
    plot_multi_order_scattering(x, J, Q, normalise_1=normalise_1, normalise_2=normalise_2)
    #signal_visualization(x, "Fast Boat", fs=2205)

    file_path = model_path2 + "simulationType_4_4.wav"
    _, x = read(file_path)
    plot_multi_order_scattering(x, J, Q, normalise_1=normalise_1, normalise_2=normalise_2)
    #signal_visualization(x, "Target Vessel", fs=2205)

    file_path = model_path2 + "simulationType_5_4.wav"
    _, x = read(file_path)
    plot_multi_order_scattering(x, J, Q, normalise_1=normalise_1, normalise_2=normalise_2)
    #signal_visualization(x, "Cetacean", fs=2205)
    """

    """
    frequency_normalisation_order_1_vectors = []
    frequency_normalisation_order_2_vectors = []
    for i in range(25):
        file_path = model_path + "simulationType_0-0-0_" + str(i) + ".wav"
        _, x = read(file_path)
        v1, v2  = normalised_frquency_vector(x, J, Q, epsilon_order_1=epsilon_order_1, epsilon_order_2=epsilon_order_2)
        frequency_normalisation_order_1_vectors.append(v1)
        frequency_normalisation_order_2_vectors.append(v2)
    frequency_normalisation_order_1_vector = np.mean(frequency_normalisation_order_1_vectors, axis=0)
    frequency_normalisation_order_2_vector = np.mean(frequency_normalisation_order_2_vectors, axis=0)
    """
main()
