# visualization/plot_plotly.py

import plotly.graph_objects as go
from collections import defaultdict
from models.states import EpidemicState

def plot_spatial(distribution_data, steps_to_visualize, width, height):
    for step in steps_to_visualize:
        fig = go.Figure()
        for state in EpidemicState:
            coords = distribution_data[step][state]
            if coords:
                x, y = zip(*coords)
            else:
                x, y = [], []
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='markers',
                name=state.name,
                marker=dict(size=6, opacity=0.6)
            ))
        fig.update_layout(
            title=f"Distribución Espacial (Paso {step})",
            xaxis_title="Coordenada X",
            yaxis_title="Coordenada Y",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            showlegend=True
        )
        fig.show()

def plot_interactive(distribution_data, max_steps, width, height):
    frames = []
    for step in range(1, max_steps+1):
        data = []
        for state in EpidemicState:
            coords = distribution_data[step][state]
            if coords:
                x, y = zip(*coords)
            else:
                x, y = [], []
            scatter = go.Scatter(
                x=x, y=y,
                mode='markers',
                name=state.name,
                marker=dict(size=6, opacity=0.6),
                visible=False
            )
            data.append(scatter)
        frames.append(go.Frame(data=data, name=str(step)))

    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            title="Distribución Espacial de Estados Epidemiológicos",
            xaxis=dict(range=[0, width], title="Coordenada X"),
            yaxis=dict(range=[0, height], title="Coordenada Y", scaleanchor="x", scaleratio=1),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[dict(label="Play",
                              method="animate",
                              args=[None, {"frame": {"duration": 300, "redraw": True},
                                           "fromcurrent": True, "transition": {"duration": 0}}])]
            )]
        ),
        frames=frames
    )

    # Añadir trazas “vacías” para que aparezcan en la leyenda
    for state in EpidemicState:
        fig.add_trace(go.Scatter(x=[], y=[], mode='markers', name=state.name))

    fig.show()
