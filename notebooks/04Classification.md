# Classification

El objetivo es clasificar la serie de tiempo en diferentes clases. Pese a que haya algo "normal", muchas veces son solo clases diferentes definidas.

# Feature-based Classification
In the easiest case, we could just use each time step t as a feature $X_t$.
This approach would have quite some advantages:
- Simplicity: Can be implemented quickly with existing algorithms.
- Interpretability: Specifically with tree-based models, important time can be found

```ad-note
Se trata cada atributo de tiempo como si fueran datos tabulares normales. Es decir, el input es el tiempo y el output sería la clasificación.
```

While this may already be the solution, the list of potential **disadvantages** is also long:
- data must already be aligned so that $X_t$ has the same meaning across instances
- if time series have different lengths, they need to be equalized in length
- potentially high dimensionality and implied risk of overfitting and expensive training
- important temporal relationships and dynamics between successive time points are ignored

```add-note
Los modelos no tienen en cuenta el orden del tiempo. (lo toman solo como un input, pero no la importancia del orden y el paso del tiempo).
```
### Classical Feature Based Classification
Si hay más de 10.000 pasos $(10^4)$, realizar la clasificación de esta forma es imposible, no se debe hacer. Se puede generar overfitting y más problemas. 

In some cases, the process has a completely different average behavior for different classes.
Such an average behavior might reflect in different distributional properties:
- average level of operation (process mean)
dispersion (process variance)
- domain of operation (min/max values)
skewness, kurtosis
- So we can summarize each time series by some of its statistics, which become the features.

##### Time-Series Classification that is (mostly) Independent of Time
The above approaches seem to make only limited use of the time information.

In the direct approach, we use classical learning algorithms, which are *agnostic to the order of
the attributes*, and the aggregation throws time away altogether.

So we could completely permute attributes (as long as we memorize which time step goes where) and get the same results. Time is only used to align instances.

**In the case of aggregating statistics, the case is even more extreme: we could even permute the time for each instance independently.**

Intuitively, both approaches seem sub-optimal.

### Transformations
Instead of working on the time series itself, we can also **transform** them before.
This means to derive a new representation of the same time series over time, which usually implies a comparable number of features necessary. We discuss two options.

**1. Discrete Fourier Transform (DFT)**: Decomposes the series into a sinusoidal component representation.

**2. Wavelet Transforms** Decomposes the series into a series of prevalence of known shapes.

#### Discrete Fourier Transform
Given a sequence of N complex numbers x0, x1, . . . , xN−1, the Discrete Fourier Transform Xk at frequency index k is defined as:

*Picture*

Since $e^{ix} = cos(x) + i*sin(x)$, this can also be written in terms of sine and cosine.

Realizar la transformación de Fourier permite mejorar el performance del modelo, sin embargo el costo de hacer esto es que se duplica la cantidad de datos.

Se hace la transformación y sobre esta se hace la clasificación de la serie.

#### Wavelets
Usa templates de patrones que yo ya conozco. La idea es tomar la plantilla y moverla en la serie de tiempo para ver que tan prevalente es en el tiempo. 

Es practicamente como hacer una convolución. Usa un kernel (filtro) y lo va aplicando sobre la serie usando un producto punto.

El producto punto calcula la similitud. Si el producto punto es 0 los vectores son perpendiculares. Si los vectores apuntan en la misma dirección obtenemos un resultado, en general, *el producto punto es la medida de similitud entre dos vectores.* 

```ad-note
Con el producto punto calculo la similitud de mi plantilla con mi serie de tiempo para cada ventana de tiempo. **Entre más alto el valor del producto punto, más prevalente es el wavelet en esa posición de la serie.**
```

Wavelet transform simply replaces the original time series by new series that measure, per time
step t, the prevalence of the wavelet at t.

One can think of the prevalence of the (dilated) wavelet function at time t as a new feature Xt.

The prevalence of the (dilated) wavelet is determined through a 1D convolution.

One (arbitrarily) distinguishes low-pass filters and high-pass filters, according to to capture
patterns with different frequencies in the time series

##### Wavelets - Discrete Wavelet Transform (DWT)
Require: x1, .., xT , Number of levels num_levels

1: Define the mother wavelet function and filter coefficients  
2: Initialize empty lists for approximation coefficients and detail coefficients  
3: for each level from 1 to num_levels do  
4: Convolve the signal with the low-pass filter to obtain approximation coefficients  
5: Convolve the signal with the high-pass filter to obtain detail coefficients  
6: Downsample the approximation coefficients  
7: Append the approximation coefficients to the list  
8: Append the detail coefficients to the list  
9: Dilate both filters (scale to double size)  
10: end for  
11: return Approximation coefficients, Detail coefficients  

#### Wavelets - Discussion
Wavelets seem to have several advantages:
- more informed features
- decomposition into potentially many patterns

```ad-important
Usar Wavelet es mejor, porque practicamente Fourier es un wavelet de seno y coseno. 
Usando wavelet puedo integrar mas información en el proceso, puedo aplicar varios sobre la misma serie. 
```

There are however also issues or debatable aspects about wavelets:
- which wavelet to choose?
- how many levels to use?
- how to downsample?
- why to wait until the dilation factor is just right for data? -> *problema de optimización para saber el factor optimo de escala.*
- why to separate low and high pass filters instead of just adjusting a set of filters?

# Classification based on Distances and Similarities
## Classification based on Distances
One algorithm we can use for classification in the above sense is kNN.

One interesting observation is that kNN uses the data only to compute a metric

$φ : X × X → R^+$

in the input space X , which is usually the Euclidean distance in X = R^d.

```ad-note
kNN calcula la distancia pero no es necesaria que sea euclidiana.
```
Now if X is not a vector space but simply a space of objects, we might still be able to somehow determine such a (non-Euclidean) distance.

In particular, time series may even have different lengths.

### Dynamic Time Warping
No siempre se debe comparar las distancias en el mismo punto de tiempo, porque la serie de tiempo puede que tenga la misma forma pero que esté corrida un tiempo.                                                                                                         It seems desirable to connect xt not with yt but with $y_{π(t)}$ so that the value yπ(t) corresponds in y what xt
is in x. Let 

$δ_{i,j}:= |x_i − y_j|$

be the difference of the values in the two series at times i and j, respectively.

We would like to connect points so that if $x_s$ is connected with $y_t$, then δs,t should be small.

```ad-note
La idea es conectar parejas en el tiempo y ver que tan diferentes son entre si. Es decir, para cada instante de tiempo de la serie 1 calcular la distancia para cada uno de los puntos de la serie 2, el resultado es una matriz de distancias.
```
To find such a mapping $π$, one solves the following optimization problem:

$arg min_{π:\{1,..,k\}→\{1,..,l\}} \sum{}_{i=1}\delta_{i,π(i)}$

such that
- π(1) = 1 and π(k) = l
- π(i − 1) ≤ π(i) ≤ π(i − 1) + 1 for all i ≥ 1

The second condition means that π must respect the ordering.

Applying this π is called **Dynamic Time Warping (DTW9)**.

The solution to the Dynamic Time Warping problem has a nice interpretation in terms of the
distance matrix as it is the cheapest path from (1, 1) to (k, l) in that matrix:

*Insert Dynamic Time Warping Heat map picture*

Se traduce en un problema de encontrar la ruta para llegar de (1,1) a (k,l) con el menor peso (la ruta tiene que pasar por el color menos oscuro), mientras mas oscura la grafica tiene un mayor valor la distancia. El problema es que si las series son muy largas tendré que calcular muchas distancias, es decir, el problema se puede crecer mucho.

The sum of the distances on this path is the DTW metric and can be used with kNN.

La ventaja es que podemos comparar series de diferentes largos de tiempo. Además, si agrego una nueva serie ya tengo mi modelo y no tengo que volver a entrenarlo.

#### Dynamic Time Warping - Discussion
kNN with DTW has some obvious advantages:
- we can do classification with variable length time series 
- we not do not need patterns to be aligned manually before 
- training kNN is for free, so extendind the DB causes no cost.

On the downside, we have the following:
- computing DTW can be notoriously expensive,
- we should not over-estimate the ability of DTW to express similarity.

Muchas veces la distancia no es una buena metrica.. porque depende de varios factores. 
No esta mirando la forma, sino solo la distancia.

```ad-note
Puedo tener la misma forma pero solo corrida en el eje Y y la distancia que me da este metodo va a ser mucha. Pese a que sean la misma serie, me dirá que no se parecen solo porque la distancia es grande.
```