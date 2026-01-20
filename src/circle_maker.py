import jax
from helpers.aerofoil_save_funcs import save

array = jax.numpy.zeros((300, 300))
for i in range(len(array)):
    for j in range(len(array[0])):
        if (i-150)**2 + (j-150)**2 <= 2500:
            array = array.at[i, j].set(1)


save(array)
print("done")