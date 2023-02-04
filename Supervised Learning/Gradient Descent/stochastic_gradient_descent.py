def SGD(f, theta0, alpha, num_iters):
    """ 
       Arguments:
       f -- the function to optimize, it takes a single argument
            and yield two outputs, a cost and the gradient
            with respect to the arguments
       theta0 -- the initial point to start SGD from
       num_iters -- total iterations to run SGD for
       Return:
       theta -- the parameter value after SGD finishes
    """
    theta = theta0
    for iter in range(num_iters):
      # For python 2.x - use xrange
        grad = f(theta)[1]
   
        # there is NO dot product ! return theta
        theta = theta - (alpha * grad) 