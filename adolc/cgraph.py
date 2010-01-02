import copy
import numpy
import numpy.testing
import wrapped_functions

class AdolcProgram(object):
    def __init__(self):
        """
        This class provides a thin wrapper of the the wrapped ADOL-C functions as hov_forward, hos_forward, etc.
        The rationale is to provide a workaround for all the quirks in ADOL-C.
        
        Using this class allows you:
            * provide a list  of input arguments xs (and optionally their corresponding directions Vs)
            * each input argument may be of arbitrary shape, e.g. xs[i].shape = (3,5,6)
            * it is possible to call a vector forward with P directions and then call a vector reverse with Q directions
            
        For more information see the doc of `AdolcProgram.forward` and `AdolcProgram.reverse`.
        """
        self.tape_tag = None
        self.independentVariableShapeList = []
        self.dependentVariableShapeList = []
        
    def trace_on(self, tape_tag):
        self.tape_tag = tape_tag
        wrapped_functions.trace_on(tape_tag)
            
    def trace_off(self):
        wrapped_functions.trace_off()
    
    def independent(self, x):
        if numpy.ndim(x) == 0:
            x = numpy.array([x])
        self.independentVariableShapeList.append(numpy.shape(x))
        wrapped_functions.independent(x)
        
    def dependent(self, x):
        if numpy.ndim(x) == 0:
            x = numpy.array([x])
        self.dependentVariableShapeList.append(numpy.shape(x))
        wrapped_functions.dependent(x)
        
    def jacobian(self, xs):
        rx_list = []
        for nx,x in enumerate(xs):
            
            numpy.testing.assert_array_almost_equal(self.independentVariableShapeList[nx], numpy.shape(x), err_msg = '\ntaped xs[%d].shape != forward xs[%d]\n'%(nx,nx))
            rx = numpy.ravel(x)
            rx_list.append(rx)
        self.x = numpy.concatenate(rx_list)
        return wrapped_functions.jacobian(self.tape_tag, self.x)
        
        
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
            if numpy.isscalar(x):
                x = numpy.asarray([x])
            numpy.testing.assert_array_almost_equal(self.independentVariableShapeList[nx], numpy.shape(x), err_msg = '\ntaped xs[%d].shape != forward xs[%d]\n'%(nx,nx))
            rx = numpy.ravel(x)
            rx_list.append(rx)
        self.x = numpy.concatenate(rx_list)
        
        if Vs != None:
            rV_list = []        
            for nV,V in enumerate(Vs):
                V_shp = numpy.shape(V)
                try:
                    numpy.testing.assert_array_almost_equal(self.independentVariableShapeList[nV], V_shp[:-2])
                except:
                    raise ValueError('taped independentVariableShapeList = %s\n but supplied Vs = %s'%(str(self.independentVariableShapeList), str(map(numpy.shape, Vs))))
                rV_list.append(numpy.reshape(V, (numpy.prod(V_shp[:-2]),) + V_shp[-2:]))
            self.V = numpy.ascontiguousarray(numpy.concatenate(rV_list,axis=0))
            
        # run the ADOL-C functions
        # ------------------------
        if Vs == None:
            self.y = wrapped_functions.zos_forward(self.tape_tag, self.x, keep=keep)
        
        else:
            N,P,D = self.V.shape
            if keep == 0:
                self.y,self.W = wrapped_functions.hov_forward(self.tape_tag, self.x, self.V)
                
            elif P == 1:
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
        Wbari is an array of shape (Q, yi.shape, P, D+1)
        Vbari is an array of shape (Q, xi.shape, P, D+1)
        """
        
        # prepare Wbar as (Q,M,P,D+1) array
        rWbar_list = []        
        for m,Wbar in enumerate(Wbars):
            Wbar_shp = numpy.shape(Wbar)
            try:
                assert len(self.dependentVariableShapeList[m])+3 == numpy.ndim(Wbar)
            except:
               raise ValueError('Wbar.shape must be (Q, yi.shape,P, D+1)=%s but provided %s'%( '(Q,'+str(self.dependentVariableShapeList[m]) +',P,D+1)' ,str(Wbar_shp)))
            try:
                assert self.dependentVariableShapeList[m] ==  Wbar_shp[1:-2]
            except:                
                raise ValueError('taped: Wbar.shape = %s but provided: Wbar.shape = %s '%(str(self.dependentVariableShapeList[m]),str(Wbar_shp[1:-2])))

            rWbar_list.append(numpy.reshape(Wbar, (Wbar_shp[0],) + (numpy.prod(Wbar_shp[1:-2]),) + Wbar_shp[-2:]))

        
        Wbar = numpy.ascontiguousarray(numpy.concatenate(rWbar_list,axis=1))
        
        if self.Vs != None:
            """ this branch is executed after a forward(xs, Vs!=None)"""
            N,PV,DV = self.V.shape
            Q,M,P,Dp1 = Wbar.shape
            D = Dp1 - 1
            try:
                assert PV == P
            except:
                raise ValueError('The number of directions P=%d in the forward run does not match the number of directions P=%d for the reverse run'%(PV,P))
            try:
                assert DV == D
            except:            
                raise ValueError('The degree D=%d in the forward run does not match the degree D=%d for the reverse run'%(DV,D))
            
            # call ADOL-C function, if P>1 repeat hos_forwards with keep then hov_ti_reverse
            # if P==1 then call direclty hov_ti_reverse
            Vbar = numpy.zeros((Q,N,P,D+1))
            for p in range(P):
                if P>=1:
                    Vtmp = self.V[:,p,:]
                    wrapped_functions.hos_forward(self.tape_tag, self.x, Vtmp, D+1)
                (Vbar[:,:,p,:],nz) = wrapped_functions.hov_ti_reverse(self.tape_tag, Wbar[:,:,p,:])
                
            
        else:
            """ this branch is executed after a forward(xs, Vs==None)"""
            Q,M,P,Dp1 = Wbar.shape
            D = Dp1 - 1
            N = self.x.size
            
            try:
                assert Dp1 == 1
            except:
                raise ValueError('You ran a forward(xs, Vs=None) and therefore in Wbar.shape =(Q,M,P,D+1) D should be 0. You provided D=%d.'%D)
                
            try:
                assert P == 1
            except:
                raise ValueError('You ran a forward(xs, Vs=None) and therefore in Wbar.shape =(Q,M,P,D+1) P should be 1. You provided P=%d.'%P)
            
            
            Vbar = numpy.zeros((Q,N,P,D+1))
            # print 'Vbar.shape',Vbar.shape
            # print 'Wbar.shape',Wbar.shape
            wrapped_functions.zos_forward(self.tape_tag, self.x, keep=1)
            Vbar[:,:,0,:] = wrapped_functions.hov_ti_reverse(self.tape_tag, Wbar[:,:,0,:])[0]
            
            
        # prepare output
        self.Vbar = Vbar
        rVbar_list = []
        count = 0
        for n, s in enumerate(self.independentVariableShapeList):
            Nx = numpy.prod(s)
            Q = Vbar.shape[0]
            rVbar_list.append(Vbar[:,count:count+Nx,:].reshape((Q,) + s+(P,Dp1,)))
            count += Nx
            
        # return output
        return rVbar_list            
            
    def __str__(self):
        return 'AdolcProgram with tape_tag=%d \n and tapestats=%s'%(self.tape_tag, str(wrapped_functions.tapestats(self.tape_tag)))
