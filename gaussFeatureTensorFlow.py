import numpy as np
import tensorflow as tf

def gaussFeature(positions):
    Ndata, Natoms, Ndim = positions.shape
    Rs = 1
    mu = 0.5
    Rc = 40
    
    with tf.Graph().as_default():
        # 1 Define tensors for the graph
        pos = tf.placeholder(tf.float64, shape=[Ndata, Natoms, Ndim])
        pos_rest = tf.slice(pos, [0, 1, 0], [Ndata, Natoms-1, Ndim])
        pos_i = tf.slice(pos, [0, 0, 0], [Ndata, 1, Ndim])
        dx = pos_rest - pos_i
        Rj = tf.reduce_sum(tf.square(dx), 2)

        # Calculate potential
        dR = Rj - Rs
        dRsq = tf.square(dR)
        h = tf.exp(-mu*dRsq / Rc**2)
        # Calculate cutoff function
        fc = tf.nn.relu(Rc - Rj) / (Rc - Rj) * 0.5*(1+tf.cos(np.pi*Rj/Rc)) 
        
        f = tf.reduce_sum(h * fc, 1)
        dfdx = tf.gradients(f, pos)
        
        
        # 2. Create session
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            # 3. execute aplication inside session
            pos_value = positions
            f_value, dfdx_value = session.run([f,dfdx], feed_dict={pos: pos_value})
            
    return f_value, dfdx_value

if __name__ == '__main__':
    Ndata = 1
    Natoms = 3
    Ndim = 2
    positions = np.arange(Ndata*Natoms*Ndim).reshape((Ndata, Natoms, Ndim))
    f, dfdx = gaussFeature(positions)
    print(positions)
    print(f)
    print(dfdx)
    
