import copy
import numpy
import numpy.testing
import wrapped_functions
class AdolcProgram(object):
    def __init__(self):
        self.tape_tag = None
        self.independentVariableShapeList = []
        self.dependentVariableShapeList = []
        
    def trace_on(self, tape_tag):
        self.tape_tag = tape_tag
        wrapped_functions.trace_on(tape_tag)
            
    def trace_off(self):
        wrapped_functions.trace_off()
    
    def independent(self, x):
        self.independentVariableShapeList.append(numpy.shape(x))
        wrapped_functions.independent(x)
        
    def dependent(self, x):
        self.dependentVariableShapeList.append(numpy.shape(x))
        wrapped_functions.dependent(x)
        
    def forward(self, xs, Vs = None,  keep = 0):
        """
        convenience function that internall calls the appropriate adolc functions
        
        generic call is:
        [y1,y2,y3,...], [W1,W2,W3,...] = self.forward([x1,x2,...],[V1,V2,...] = None, 0 <= keep <= D+1)
        where
        x_ is an array of arbitrary shape shp_xi
        and Vi the corresponding direction tensor of shape :  shp_xi + (P,D)
        
        also the outputs are in the same format as handed to the self.dependent function.
        
        forward stores the inputs in self.xs and self.Vs as well as self.ys, self.Ws
        
        """
        
        # save inputs
        xs = copy.deepcopy(xs)
        Vs = copy.deepcopy(Vs)
        self.xs = xs
        self.Vs = Vs
        
        # prepare xs and Vs
        # -----------------
        rx_list = []
        for nx,x in enumerate(xs):
            numpy.testing.assert_array_almost_equal(self.independentVariableShapeList[nx], numpy.shape(x))
            rx = numpy.ravel(x)
            rx_list.append(rx)
        self.x = numpy.ravel(rx_list)
        
        if Vs != None:
            rV_list = []        
            for nV,V in enumerate(Vs):
                V_shp = numpy.shape(V)
                numpy.testing.assert_array_almost_equal(self.independentVariableShapeList[nV], V_shp[:-2])
                rV_list.append(numpy.reshape(V, (numpy.prod(V_shp[:-2]),) + V_shp[-2:]))
            self.V = numpy.ascontiguousarray(numpy.concatenate(rV_list,axis=0))
            
        # run the ADOL-C functions
        # ------------------------
        if Vs == None:
            self.y = wrapped_functions.zos_forward(self.tape_tag, self.x, keep=keep)
        
        else:
            N,P,D = self.V.shape
            if P >= 1 and keep == 0:
                self.y,self.W = wrapped_functions.hov_forward(self.tape_tag, self.x, self.V)
                
            elif P == 1 and keep > 0:
                Vtmp = self.V.reshape((N,D))
                self.y,Wtmp = wrapped_functions.hos_forward(self.tape_tag, self.x, Vtmp, keep)
                M = Wtmp.shape[0]
                self.W = Wtmp.reshape((M,P,D))
                
            elif P > 1 and keep > 0:
                raise NotImplementedError('ADOL-C doesn\'t support higher order vector forward with keep!\n \
                    workaround: several runs forward with P=1')                      
                
        # prepare outputs
        # ---------------
        self.ys = []
        count = 0
        for ns, s in enumerate(self.dependentVariableShapeList):
            M_ns = numpy.prod(s)
            self.ys.append(self.y[count:count+M_ns].reshape(s))
            count += M_ns
        
        if Vs != None:
            self.Ws = []
            count = 0
            for ns, s in enumerate(self.dependentVariableShapeList):
                M_ns = numpy.prod(s)
                self.Ws.append(self.W[count:count+M_ns,:,:].reshape(s+(P,D)))
                count += M_ns
                
        # return outputs
        # --------------
        if Vs == None:
            return self.ys
        else:
            return (self.ys, self.Ws)                


    def reverse(self, Wbars):
        """
        convenience function that internall calls the appropriate adolc functions
        this function can only be called after self.forward has been called with keep > 0
        
        generic call is:
        [Vbar1,Vbar2,...] = self.reverse([Wbar1,Wbar2,...])
        where
        Wbari is an array of shape (Q, yi.shape, D+1)
        Vbari is an array of shape (Q, xi.shape, D+1)
        """
        
        # prepare Wbar
        rWbar_list = []        
        for m,Wbar in enumerate(Wbars):
            Wbar_shp = numpy.shape(Wbar)
            numpy.testing.assert_array_almost_equal(self.dependentVariableShapeList[m], Wbar_shp[1:-1])
            rWbar_list.append(numpy.reshape(Wbar, (Wbar_shp[0],) + (numpy.prod(Wbar_shp[1:-1]),) + (Wbar_shp[-1],)))
        Wbar = numpy.ascontiguousarray(numpy.concatenate(rWbar_list,axis=1))
        
        # call ADOL-C function
        (Vbar,nz) = wrapped_functions.hov_ti_reverse(self.tape_tag, Wbar)

        # prepare output
        rVbar_list = []
        count = 0
        for n, s in enumerate(self.independentVariableShapeList):
            Nx = numpy.prod(s)
            Q = Vbar.shape[0]
            D = Vbar.shape[-1] - 1
            rVbar_list.append(Vbar[:,count:count+Nx,:].reshape((Q,) + s+(D+1,)))
            count += Nx
            
        # return output
        return rVbar_list
        
            
    def __str__(self):
        return 'AdolcProgram with tape_tag=%d \n and tapestats=%s'%(self.tape_tag, str(wrapped_functions.tapestats(self.tape_tag)))
