import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot
import plotly.express as px
import pandas as pd

def PlotSingle(y,x=None,w=6,h=4,lw=1,ms=1,mrkr=None,
               title='', xname='x', yname='y', data='Data',
               pltly=True, save=False, file_name='plot.png'):
    if x== None: x=np.arange(0,len(y),1)
    if pltly:
        fig=make_subplots(rows=1,cols=1)
        trace = go.Scatter(x=x,y=y, name=data)
        fig.add_trace(trace, row=1, col=1)
        fig.update_layout(width = w*100, height = h*100, title = title)
        fig.update_xaxes(title_text=xname, row = 1, col = 1)
        fig.update_yaxes(title_text=yname, row = 1, col = 1)
        fig.show()
        if save: fig.write_image(file_name, scale=5)
    else:
        plt.figure(figsize=(w, h))  # Define o tamanho do gráfico
        plt.plot(x, y, linestyle='-', linewidth = lw, marker=mrkr, markersize=ms, color='b', label=data)  # Plota os dados
        plt.xlabel(xname)  # Nome do eixo X
        plt.ylabel(yname)  # Nome do eixo Y
        plt.title(title)  # Define o título do gráfico
        plt.grid(False)  # Adiciona grade ao gráfico
        plt.legend()  # Exibe legenda
        plt.show()  # Mostra o gráfico
        if save: plt.savefig(file_name, dpi=500)

def PlotSeries(y_arrays, x_arrays=None, w=8, h=5, lw=2, ms=2, mrkr=None,
               title='', xname='x', yname='y', legend_labels=None,
               pltly=True, save=False, file_name='plot.png', return_fig=False):
    """
    Plota ou cria um objeto de figura com múltiplas séries de dados.
    
    Se return_fig=True, a função retorna o objeto da figura em vez de exibi-lo.
    """
    
    if x_arrays is None:
        x_arrays = [np.arange(len(y)) for y in y_arrays]
    if legend_labels is None:
        legend_labels = [f'Série {i+1}' for i in range(len(y_arrays))]
    x_arrays = np.flip(np.array(x_arrays))
    y_arrays = np.flip(np.array(y_arrays))
    legend_labels = np.flip(np.array(legend_labels))
    dash_ = np.array([None,'dash'])
    if pltly:
        # Para Plotly, retornamos os traços (dados) e o layout (configurações) separadamente
        traces = []
        for x, y, name, dash in zip(x_arrays, y_arrays, legend_labels,dash_):
            trace = go.Scatter(x=x, y=y, name=name, mode='lines+markers',
                               line=dict(width=lw,dash=dash), marker=dict(size=ms))
            traces.append(trace)
        
        layout = go.Layout(width=w*100, height=h*100, title=title,
                           xaxis_title=xname, yaxis_title=yname,
                           legend_title_text='Legenda')
        
        fig = go.Figure(data=traces, layout=layout)
        
        if return_fig:
            return fig # Retorna o objeto completo da figura

        fig.show()
        if save:
            try:
                fig.write_image(file_name, scale=5)
                print(f"Gráfico salvo como '{file_name}'")
            except ValueError as e:
                print(f"Erro ao salvar a imagem: {e}")
                print("Instale 'kaleido': pip install -U kaleido")
    
    else: # Matplotlib
        fig, ax = plt.subplots(figsize=(w, h))
        linestyle_ = np.array(['-','--'])
        for x, y, label,linestyle in zip(x_arrays, y_arrays, legend_labels, linestyle_):
            ax.plot(x, y, linestyle=linestyle, linewidth=lw, marker=mrkr, markersize=ms, label=label)
            
        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
        
        if return_fig:
            return fig # Retorna a figura para composição

        if save:
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
            print(f"Gráfico salvo como '{file_name}'")

        plt.show()
        plt.close(fig) # Fecha a figura para liberar memória

def Plot2Axis(y_arrays, x_arrays=None, w=8, h=5, lw=1.25, ms=2,
               title='', xname='x', yname='Y2', legend_labels=None,
                 save=False, file_name='plot.png', return_fig=False):
    """
    Plots multiple data series, with an option for a secondary y-axis.

    - y_arrays1 (list of lists): Data series for the primary (left) y-axis.
    - x_arrays (list of lists): Corresponding x-values for y_arrays1.
    - y_array2 (list/array): Optional single data series for the secondary (right) y-axis.
    - yname2 (str): Label for the secondary y-axis.
    - legend_label2 (str): Legend entry for the secondary series.
    
    If return_fig=True, the function returns the figure object instead of displaying it.
    """
    yname2=r'Wind Speed $(m/s)$'
    legend_label2='Wind Speed'
    if x_arrays is None:
        x_arrays = [np.arange(len(y)) for y in y_arrays]
    if legend_labels is None:
        legend_labels = [f'Série {i+1}' for i in range(len(y_arrays))]
    

    legend_labels_flipped = np.flip(np.array(legend_labels))
    
    fig, ax1 = plt.subplots(figsize=(w, h))
    all_lines = [] # To store line objects for the unified legend
    
    # Plot primary series
    linestyle_ = ['-', ':']
    clrs = ['lightsalmon','red']
    # Plot secondary series if it exists
   
    color2 = 'black'
    # Assumes y_array2 shares the x-axis of the first primary series
    x_for_y2 = x_arrays[0]
    line2 = ax1.plot(x_for_y2, y_arrays[-1], color='black', linestyle='-',
                        linewidth=lw*1.75, label=legend_label2)
    ax1.set_ylabel(yname2, color=color2)
    ax1.tick_params(axis='y', labelcolor=color2)
    all_lines.extend(line2)

    ax2 = ax1.twinx()
    y_arrays = y_arrays[:-1]
    for i, (x, y, label) in enumerate(zip(x_arrays, y_arrays, legend_labels_flipped)):
        line = ax2.plot(x, y, linestyle=linestyle_[i], 
                        linewidth=lw*1.75, marker=None, markersize=ms, label=label, color=clrs[i])
        all_lines.extend(line)
        ax2.set_ylabel(yname,color='red')
        ax2.tick_params(axis='y', labelcolor='red')
    
    ax1.set_xlabel(xname)
    
    ax1.grid(True, which='major', linestyle='--', linewidth=0.5)


    ax1.set_title(title)
    # Create a single legend for all series
    labels = [l.get_label() for l in all_lines]
    ax2.legend(all_lines, labels, loc='best')
    
    fig.tight_layout()

    if return_fig:
        return fig
    
    if save:
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo como '{file_name}'")

    plt.show()
    plt.close(fig)

def Multi2Plot(plots_data, rows, cols, main_title='', lw=1,
              save=False, file_name='multiplot.png', fig_size=(12, 8)):
    """
    Combina múltiplos gráficos gerados pela PlotSeries em uma única imagem.

    Args:
        plots_data (list of dict): Uma lista de dicionários, onde cada dicionário
                                   contém os argumentos para uma chamada da PlotSeries.
        rows (int): Número de linhas na grade de subplots.
        cols (int): Número de colunas na grade de subplots.
        pltly (bool, optional): Define se usará Plotly ou Matplotlib. Defaults to True.
        main_title (str, optional): Título principal para o conjunto de gráficos.
        save (bool, optional): Se True, salva a imagem final. Defaults to False.
        file_name (str, optional): Nome do arquivo para salvar.
        fig_size (tuple, optional): Tamanho total da figura (para Matplotlib).
    """
    yname2=r'Wind Speed $(m/s)$'
    legend_label2='Wind Speed'
    linestyle_ = ['-', ':']
    clrs = ['lightsalmon','red']
    color2 = 'black'
    
    if len(plots_data) > rows * cols:
        print(f"Aviso: Você tem {len(plots_data)} gráficos para plotar, mas a grade é de {rows}x{cols}. Alguns gráficos não serão exibidos.")

    
    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    # Garante que 'axes' seja sempre um array iterável
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, ax1 in enumerate(axes):
        all_lines = []
        if i >= len(plots_data):
            ax1.axis('off') # Esconde eixos de subplots não utilizados
            continue

        plot_args = plots_data[i]
        
        # Desempacota os argumentos para o plot
        x_arrays = plot_args.get('x_arrays')
        y_arrays = plot_args.get('y_arrays')
        legend_labels = plot_args.get('legend_labels')
        
        # Lógica de criação de dados padrão (caso não sejam fornecidos)
        if y_arrays is None: continue
        if x_arrays is None:
            x_arrays = [np.arange(len(y)) for y in y_arrays]
        if legend_labels is None:
            legend_labels = [f'Série {j+1}' for j in range(len(y_arrays))]

        line2 = ax1.plot(x_arrays[0], y_arrays[-1], color='black', linestyle='-',
                        linewidth=lw*1.75, label=legend_label2)
        ax1.set_ylabel(yname2, color=color2)
        ax1.tick_params(axis='y', labelcolor=color2)
        all_lines.extend(line2)

        ax2 = ax1.twinx()
        y_arrays = y_arrays[:-1]
        
        for i, (x, y, label) in enumerate(zip(x_arrays, y_arrays, legend_labels)):

            line = ax2.plot(x, y, linestyle=linestyle_[i], 
                            linewidth=lw*1.75, label=label, color=clrs[i])
            all_lines.extend(line)
            ax2.tick_params(axis='y', labelcolor='red')
        
        ax1.set_title(plot_args.get('title', f'Gráfico {i+1}'))
        ax1.set_xlabel(plot_args.get('xname', 'x'))
        ax2.set_ylabel(plot_args.get('yname', 'y'),color='red')
        #if i > 4: ax.set_xlim(1000, len(x))
        ax1.grid(True)
        labels = [l.get_label() for l in all_lines]
        ax2.legend(all_lines, labels, loc='best')
        
    fig.suptitle(main_title, fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajusta para o título principal caber

    if save:
        plt.savefig(file_name, dpi=300)
        print(f"Multi-plot salvo como '{file_name}'")

    plt.show()

def MultiPlot(plots_data, rows, cols, pltly=True, main_title='', lw=1.75,
              save=False, file_name='multiplot.png', fig_size=(12, 8)):
    """
    Combina múltiplos gráficos gerados pela PlotSeries em uma única imagem.

    Args:
        plots_data (list of dict): Uma lista de dicionários, onde cada dicionário
                                   contém os argumentos para uma chamada da PlotSeries.
        rows (int): Número de linhas na grade de subplots.
        cols (int): Número de colunas na grade de subplots.
        pltly (bool, optional): Define se usará Plotly ou Matplotlib. Defaults to True.
        main_title (str, optional): Título principal para o conjunto de gráficos.
        save (bool, optional): Se True, salva a imagem final. Defaults to False.
        file_name (str, optional): Nome do arquivo para salvar.
        fig_size (tuple, optional): Tamanho total da figura (para Matplotlib).
    """
    if len(plots_data) > rows * cols:
        print(f"Aviso: Você tem {len(plots_data)} gráficos para plotar, mas a grade é de {rows}x{cols}. Alguns gráficos não serão exibidos.")

    if pltly:
        # Pega os títulos dos subplots dos dados, se existirem
        subplot_titles = [p.get('title', '') for p in plots_data]
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

        for i, plot_args in enumerate(plots_data):
            if i >= rows * cols: break
            
            # Posição na grade
            row = i // cols + 1
            col = i % cols + 1
            
            # Gera a figura temporária para extrair os dados (traços)
            temp_fig = PlotSeries(pltly=True, return_fig=True, **plot_args)
            
            # Adiciona os traços da figura temporária ao subplot correto
            for trace in temp_fig.data:
                fig.add_trace(trace, row=row, col=col)
            
            # Atualiza os eixos do subplot
            fig.update_xaxes(title_text=plot_args.get('xname', 'x'), row=row, col=col)
            fig.update_yaxes(title_text=plot_args.get('yname', 'y'), row=row, col=col)

        fig.update_layout(title_text=main_title, height=fig_size[1]*100, width=fig_size[0]*100)
        fig.show()

        if save:
            fig.write_image(file_name, scale=3)
            print(f"Multi-plot salvo como '{file_name}'")

    else: # Matplotlib
        fig, axes = plt.subplots(rows, cols, figsize=fig_size)
        # Garante que 'axes' seja sempre um array iterável
        if rows * cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i >= len(plots_data):
                ax.axis('off') # Esconde eixos de subplots não utilizados
                continue

            plot_args = plots_data[i]
            
            # Desempacota os argumentos para o plot
            x_arrays = plot_args.get('x_arrays')
            y_arrays = plot_args.get('y_arrays')
            legend_labels = plot_args.get('legend_labels')
            
            # Lógica de criação de dados padrão (caso não sejam fornecidos)
            if y_arrays is None: continue
            if x_arrays is None:
                x_arrays = [np.arange(len(y)) for y in y_arrays]
            if legend_labels is None:
                legend_labels = [f'Série {j+1}' for j in range(len(y_arrays))]
            linestyle_ = np.array(['-','--'])
            # Plota os dados no eixo (ax) correto
            for x, y, label, linestyle in zip(x_arrays, y_arrays, legend_labels, linestyle_):
                ax.plot(x, y, label=label,linestyle=linestyle, linewidth=lw)
            
            ax.set_title(plot_args.get('title', f'Gráfico {i+1}'))
            ax.set_xlabel(plot_args.get('xname', 'x'))
            ax.set_ylabel(plot_args.get('yname', 'y'))
            #if i > 4: ax.set_xlim(1000, len(x))
            ax.grid(True)
            ax.legend()
            
        fig.suptitle(main_title, fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajusta para o título principal caber

        if save:
            plt.savefig(file_name, dpi=300)
            print(f"Multi-plot salvo como '{file_name}'")

        plt.show()



def AllPlots(dTrain,dTrain2):
    all_plots= [
        {
            'y_arrays': [dTrain.PG,dTrain2.PG],
            'x_arrays': None,
            'title': 'Generated Power',
            'xname': 'Sample',
            'yname': r'$P_{G}\ (W)$',
            'legend_labels': ['w/ RUL Control','w/o RUL Control']
        },
        {
            'y_arrays': [dTrain.PD,dTrain2.PD],
            'x_arrays': None,
            'title': 'Dissipated Power',
            'xname': None,
            'yname': r'$P_{D}\ (W)$',
            'legend_labels':['w/ RUL Control','w/o RUL Control']
        },
        {
            'y_arrays': [dTrain.EG,dTrain2.EG],
            'x_arrays': None,
            'title': 'Cumulative Generated Energy',
            'xname': None,
            'yname': r'$E_{G}\ (Ws)$',
            'legend_labels': ['w/ RUL Control','w/o RUL Control']
        },
        {
            'y_arrays': [dTrain.ED,dTrain2.ED],
            'x_arrays': None,
            'title': 'Cumulative Dissipated Energy',
            'xname': None,
            'yname': r'$E_{D}\ (Ws)$',
            'legend_labels': ['w/ RUL Control','w/o RUL Control']
        },
        {
            'y_arrays': [dTrain.K_mppt[1:],dTrain2.K_mppt[1:]],
            'x_arrays': None,
            'title': r'$K_{mppt}$ gain consecutively',
            'xname': None,
            'yname': r'$K_{mppt}$ gain',
            'legend_labels': ['w/ RUL Control','w/o RUL Control']
        },
        {
            'y_arrays': [dTrain.X_[:,0],dTrain2.X_[:,0]],
            'x_arrays': None,
            'title': r'$\omega_{r}$ consecutively',
            'xname': None,
            'yname': r'$\omega_{r}$ (rad/s)',
            'legend_labels': ['w/ RUL Control','w/o RUL Control']
        },
        {
            'y_arrays': [dTrain.X_[:,1],dTrain2.X_[:,1]],
            'x_arrays': None,
            'title': r'$\omega_{g}$ consecutively',
            'xname': None,
            'yname': r'$\omega_{g}$ (rad/s)',
            'legend_labels': ['w/o RUL Control','w/o RUL Control']
        },
        {
            'y_arrays': [dTrain.X_[:,2],dTrain2.X_[:,2]],
            'x_arrays': None,
            'title': r'$\theta_{ts}$ consecutively',
            'xname': None,
            'yname': r'$\theta_{ts}$ (rad)',
            'legend_labels': ['w/ RUL Control','w/o RUL Control']
        },
        {
            'y_arrays': [dTrain.dX_[:,0],dTrain2.dX_[:,0]],
            'x_arrays': None,
            'title': r'$\dot{\omega}_{r}$ consecutively',
            'xname': None,
            'yname': r'$\dot{\omega}_{r}$ (rad/s)',
            'legend_labels': ['w/ RUL Control','w/o RUL Control']
        },
        {
            'y_arrays': [dTrain.dX_[:,1],dTrain2.dX_[:,1]],
            'x_arrays': None,
            'title': r'$\dot{\omega}_{g}$ consecutively',
            'xname': None,
            'yname': r'$\dot{\omega}_{g}$ (rad/s)',
            'legend_labels': ['w/ RUL Control','w/o RUL Control']
        },
        {
            'y_arrays': [dTrain.dX_[:,2],dTrain2.dX_[:,2]],
            'x_arrays': None,
            'title': r'$\dot{\theta}_{ts}$ consecutively',
            'xname': None,
            'yname': r'$\dot{\theta}_{ts}$ (rad)',
            'legend_labels': ['w/ RUL Control','w/o RUL Control']
        },
        {
            'y_arrays': [dTrain.tR,dTrain2.tR],
            'x_arrays': None,
            'title': r'${\tau}_{r}$ consecutively',
            'xname': None,
            'yname': r'${\tau}_{r}$ (N.m)',
            'legend_labels': ['w/ RUL Control','w/o RUL Control']
        },
        {
            'y_arrays': [dTrain.tG,dTrain2.tG],
            'x_arrays': None,
            'title': r'${\tau}_{g}$ consecutively',
            'xname': None,
            'yname': r'${\tau}_{g}$ (N.m)',
            'legend_labels': ['w/ RUL Control','w/o RUL Control']
        },
        ]
    
    return all_plots

def AllPlots2(dTrain,dTrain2,v):
    all_plots= [
        {
            'y_arrays': [dTrain.PG,dTrain2.PG,v],
            'x_arrays': None,
            'title': 'Generated Power',
            'xname': 'Sample',
            'yname': r'$P_{G}\ (W)$',
            'legend_labels': ['w/o RUL Control','w/ RUL Control']
        },
        {
            'y_arrays': [dTrain.PD,dTrain2.PD,v],
            'x_arrays': None,
            'title': 'Dissipated Power',
            'xname': None,
            'yname': r'$P_{D}\ (W)$',
            'legend_labels':['w/o RUL Control','w/ RUL Control']
        },
        {
            'y_arrays': [dTrain.EG,dTrain2.EG,v],
            'x_arrays': None,
            'title': 'Cumulative Generated Energy',
            'xname': None,
            'yname': r'$E_{G}\ (Ws)$',
            'legend_labels': ['w/o RUL Control','w/ RUL Control']
        },
        {
            'y_arrays': [dTrain.ED,dTrain2.ED,v],
            'x_arrays': None,
            'title': 'Cumulative Dissipated Energy',
            'xname': None,
            'yname': r'$E_{D}\ (Ws)$',
            'legend_labels': ['w/o RUL Control','w/ RUL Control']
        },
        {
            'y_arrays': [dTrain.K_mppt[1:],dTrain2.K_mppt[1:],v],
            'x_arrays': None,
            'title': r'$K_{mppt}$ gain consecutively',
            'xname': None,
            'yname': r'$K_{mppt}$ gain',
            'legend_labels': ['w/o RUL Control','w/ RUL Control']
        },
        {
            'y_arrays': [dTrain.X_[1:,0],dTrain2.X_[:,0],v],
            'x_arrays': None,
            'title': r'$\omega_{r}$ consecutively',
            'xname': None,
            'yname': r'$\omega_{r}$ (rad/s)',
            'legend_labels': ['w/o RUL Control','w/ RUL Control']
        },
        {
            'y_arrays': [dTrain.X_[1:,1],dTrain2.X_[:,1],v],
            'x_arrays': None,
            'title': r'$\omega_{g}$ consecutively',
            'xname': None,
            'yname': r'$\omega_{g}$ (rad/s)',
            'legend_labels': ['w/o RUL Control','w/ RUL Control']
        },
        {
            'y_arrays': [dTrain.X_[1:,2],dTrain2.X_[:,2],v],
            'x_arrays': None,
            'title': r'$\theta_{ts}$ consecutively',
            'xname': None,
            'yname': r'$\theta_{ts}$ (rad)',
            'legend_labels': ['w/o RUL Control','w/ RUL Control']
        },
        {
            'y_arrays': [dTrain.dX_[1:,0],dTrain2.dX_[:,0],v],
            'x_arrays': None,
            'title': r'$\dot{\omega}_{r}$ consecutively',
            'xname': None,
            'yname': r'$\dot{\omega}_{r}$ (rad/s)',
            'legend_labels': ['w/o RUL Control','w/ RUL Control']
        },
        {
            'y_arrays': [dTrain.dX_[1:,1],dTrain2.dX_[:,1],v],
            'x_arrays': None,
            'title': r'$\dot{\omega}_{g}$ consecutively',
            'xname': None,
            'yname': r'$\dot{\omega}_{g}$ (rad/s)',
            'legend_labels': ['w/o RUL Control','w/ RUL Control']
        },
        {
            'y_arrays': [dTrain.dX_[1:,2],dTrain2.dX_[:,2],v],
            'x_arrays': None,
            'title': r'$\dot{\theta}_{ts}$ consecutively',
            'xname': None,
            'yname': r'$\dot{\theta}_{ts}$ (rad)',
            'legend_labels': ['w/o RUL Control','w/ RUL Control']
        },
        ]
    
    return all_plots

