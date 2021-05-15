# Script to generate plots of confidence interval results
# Intended to be run from within RStudio; does not save plots
# (could be easily ammended to automatically save plots as images)
# Author: Alene Rhea, May 2021

# BEFORE RUNNING THIS SCRIPT, MUST RUN utils/combineResults.py
# This will generate an updated 'all_ci_mse_simple.csv' file


library(ggplot2)
library(gridExtra)
library(lemon)
library(here)

# Read in CI and MSE data
res <- read.csv(here('results','all_ci_mse_simple.csv'))

# Add combined meta-base column
res$meta.base <- paste(res$meta, res$base, ' ')

# Base plotting function
ci_subplot <- function(data, x, shapes, colors, xjitter, 
                       xmin, xmax, nom, nom.x, nom.raise, label.x, label){
  
  if(length(unique(data$meta.base))==3){
    shapes <- shapes[rep(c(TRUE,FALSE),3)]
  }
  
  g <- ggplot(data=data, aes(x=data[,x], y=coverage, color=ci_type))+
    geom_hline(aes(yintercept=0.95), linetype='dashed', size=0.25)+
    geom_jitter(aes(shape=meta.base), size=2.5, alpha=0.5, stroke=1,
                width=xjitter, height=0.01)+
    scale_shape_manual(values = shapes)+
    scale_color_manual(values = colors)+
    xlim(xmin,xmax)+
    ylim(-0.01,1.05)+
    ylab('Coverage')+
    guides(shape=guide_legend(title="Learners"),
           color=guide_legend(title="CI Method"))+
    geom_label(x=label.x, y=0.625, label=label, size=3.5, color='black')
  
  if (nom){
    g <- g + annotate("text", x=label.x, y=1.02,
                      label='Nominal Coverage: 95%', size=3.33)
  }
  
  
  if(x=='B'){
    return (g + geom_line(aes(shape=meta.base), size=0.4, alpha=0.4) + 
              xlab('Bootstrap Samples (B)'))
  } else if(x=='mse_samp1'){
    return (g + xlab('Sample MSE'))
  }else if(x=='mean_length'){
    return (g + xlab('Mean CI Length'))
  }
  
}


# Plot 1 subplot for E and 1 subplot for F
plot_ci <- function(data, x, B=FALSE, nom.raise=0){
  
  # Create plot labels
  label.e <- paste("Simulation", "E")
  label.f <- paste("Simulation", "F")
  if (B){
      # Subset the data if specific B value passed
      data <- data[data$B==B,]
      label.e = paste(label.e, '\nB = ', B, sep='')
      label.f = paste(label.f, '\nB = ', B, sep='')
  }

  xjitter <- .02*max(data[,x])
  xmax <- max(data[,x])+xjitter
  xmin <- min(0, min(data[,x])-xjitter)
  label.x <- (xmax-xmin)/2
  
  data.e <- data[data$sim=='E',]
  data.f <- data[data$sim=='F',]
  
  shapes=c(16,21,17,24,15,22)
  colors=c('cornflowerblue', 'goldenrod1', 'springgreen4', 
           'firebrick3', 'mediumpurple1')
  
  e <- ci_subplot(data=data.e, x=x, label=label.e, nom=FALSE,
                  shapes=shapes, colors=colors,  
                  xjitter=xjitter, xmin=xmin, xmax=xmax, 
                  nom.raise=nom.raise, label.x=label.x)
  f <- ci_subplot(data=data.f, x=x, label=label.f, nom=TRUE,
                  shapes=shapes, colors=colors,
                  xjitter=xjitter, xmin=xmin, xmax=xmax, 
                  nom.raise=nom.raise, label.x=label.x)
  
  legend <- g_legend(e +  theme(legend.title=element_text(size=10),
                              legend.text = element_text(size=9),
                              legend.spacing = unit(0.04, "cm"),
                              legend.box="horizontal"))
  
  ef.grid <- grid.arrange(e+theme(legend.position = 'hidden'), 
                          f+ylab('')+theme(legend.position = 'hidden'), 
                          legend, widths=c(2, 2, 2), nrow=1)
  
  return(ef.grid)
}

# Plot 1K Length
plot_ci(data=res, x='mean_length', B=1000)

# Plot all with B
plot_ci(data=res, x='B', nom.raise=0.04)

# Plot 1K MSE
plot_ci(data=res, x='mse_samp1', B=1000)

# Plot 10K Length
plot_ci(data=res, x='mean_length', B=10000)