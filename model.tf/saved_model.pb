Û
®
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
=
RGBToHSV
images"T
output"T"
Ttype0:
2
@
ReadVariableOp
resource
value"dtype"
dtypetype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	

ResizeBilinear
images"T
size
resized_images"
Ttype:
2	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8Ò
~
conv1a/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1a/kernel
w
!conv1a/kernel/Read/ReadVariableOpReadVariableOpconv1a/kernel*&
_output_shapes
:*
dtype0
n
conv1a/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1a/bias
g
conv1a/bias/Read/ReadVariableOpReadVariableOpconv1a/bias*
_output_shapes
:*
dtype0
~
conv1b/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1b/kernel
w
!conv1b/kernel/Read/ReadVariableOpReadVariableOpconv1b/kernel*&
_output_shapes
:*
dtype0
n
conv1b/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1b/bias
g
conv1b/bias/Read/ReadVariableOpReadVariableOpconv1b/bias*
_output_shapes
:*
dtype0
~
conv2a/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2a/kernel
w
!conv2a/kernel/Read/ReadVariableOpReadVariableOpconv2a/kernel*&
_output_shapes
: *
dtype0
n
conv2a/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2a/bias
g
conv2a/bias/Read/ReadVariableOpReadVariableOpconv2a/bias*
_output_shapes
: *
dtype0
~
conv2b/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *
shared_nameconv2b/kernel
w
!conv2b/kernel/Read/ReadVariableOpReadVariableOpconv2b/kernel*&
_output_shapes
:  *
dtype0
n
conv2b/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2b/bias
g
conv2b/bias/Read/ReadVariableOpReadVariableOpconv2b/bias*
_output_shapes
: *
dtype0
~
conv2c/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameconv2c/kernel
w
!conv2c/kernel/Read/ReadVariableOpReadVariableOpconv2c/kernel*&
_output_shapes
: **
dtype0
n
conv2c/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2c/bias
g
conv2c/bias/Read/ReadVariableOpReadVariableOpconv2c/bias*
_output_shapes
:**
dtype0
x
dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense1/kernel
q
!dense1/kernel/Read/ReadVariableOpReadVariableOpdense1/kernel* 
_output_shapes
:
*
dtype0
o
dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense1/bias
h
dense1/bias/Read/ReadVariableOpReadVariableOpdense1/bias*
_output_shapes	
:*
dtype0
w
dense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense2/kernel
p
!dense2/kernel/Read/ReadVariableOpReadVariableOpdense2/kernel*
_output_shapes
:	*
dtype0
n
dense2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense2/bias
g
dense2/bias/Read/ReadVariableOpReadVariableOpdense2/bias*
_output_shapes
:*
dtype0
v
dense3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense3/kernel
o
!dense3/kernel/Read/ReadVariableOpReadVariableOpdense3/kernel*
_output_shapes

:*
dtype0
n
dense3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense3/bias
g
dense3/bias/Read/ReadVariableOpReadVariableOpdense3/bias*
_output_shapes
:*
dtype0

NoOpNoOp
×.
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*.
value.B. Bþ-
é
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
	optimizer
loss
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
 	variables
!regularization_losses
"trainable_variables
#	keras_api
R
$	variables
%regularization_losses
&trainable_variables
'	keras_api
h

(kernel
)bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
h

.kernel
/bias
0	variables
1regularization_losses
2trainable_variables
3	keras_api
h

4kernel
5bias
6	variables
7regularization_losses
8trainable_variables
9	keras_api
R
:	variables
;regularization_losses
<trainable_variables
=	keras_api
h

>kernel
?bias
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
h

Dkernel
Ebias
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
h

Jkernel
Kbias
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
 
 
 
v
0
1
2
3
(4
)5
.6
/7
48
59
>10
?11
D12
E13
J14
K15
v
0
1
2
3
(4
)5
.6
/7
48
59
>10
?11
D12
E13
J14
K15
­
Pnon_trainable_variables
regularization_losses

Qlayers
Rlayer_regularization_losses
Smetrics
Tlayer_metrics
trainable_variables
	variables
 
 
 
 
­
Unon_trainable_variables
	variables
regularization_losses

Vlayers
Wmetrics
Xlayer_metrics
trainable_variables
Ylayer_regularization_losses
YW
VARIABLE_VALUEconv1a/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv1a/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
Znon_trainable_variables
	variables
regularization_losses

[layers
\metrics
]layer_metrics
trainable_variables
^layer_regularization_losses
YW
VARIABLE_VALUEconv1b/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv1b/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
_non_trainable_variables
 	variables
!regularization_losses

`layers
ametrics
blayer_metrics
"trainable_variables
clayer_regularization_losses
 
 
 
­
dnon_trainable_variables
$	variables
%regularization_losses

elayers
fmetrics
glayer_metrics
&trainable_variables
hlayer_regularization_losses
YW
VARIABLE_VALUEconv2a/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2a/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
 

(0
)1
­
inon_trainable_variables
*	variables
+regularization_losses

jlayers
kmetrics
llayer_metrics
,trainable_variables
mlayer_regularization_losses
YW
VARIABLE_VALUEconv2b/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2b/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1
 

.0
/1
­
nnon_trainable_variables
0	variables
1regularization_losses

olayers
pmetrics
qlayer_metrics
2trainable_variables
rlayer_regularization_losses
YW
VARIABLE_VALUEconv2c/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2c/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

40
51
 

40
51
­
snon_trainable_variables
6	variables
7regularization_losses

tlayers
umetrics
vlayer_metrics
8trainable_variables
wlayer_regularization_losses
 
 
 
­
xnon_trainable_variables
:	variables
;regularization_losses

ylayers
zmetrics
{layer_metrics
<trainable_variables
|layer_regularization_losses
YW
VARIABLE_VALUEdense1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

>0
?1
 

>0
?1
¯
}non_trainable_variables
@	variables
Aregularization_losses

~layers
metrics
layer_metrics
Btrainable_variables
 layer_regularization_losses
YW
VARIABLE_VALUEdense2/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense2/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

D0
E1
 

D0
E1
²
non_trainable_variables
F	variables
Gregularization_losses
layers
metrics
layer_metrics
Htrainable_variables
 layer_regularization_losses
YW
VARIABLE_VALUEdense3/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense3/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

J0
K1
 

J0
K1
²
non_trainable_variables
L	variables
Mregularization_losses
layers
metrics
layer_metrics
Ntrainable_variables
 layer_regularization_losses
 
V
0
1
2
3
4
5
6
7
	8

9
10
11
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

serving_default_input_2Placeholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿ À
¶
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2conv1a/kernelconv1a/biasconv1b/kernelconv1b/biasconv2a/kernelconv2a/biasconv2b/kernelconv2b/biasconv2c/kernelconv2c/biasdense1/kerneldense1/biasdense2/kerneldense2/biasdense3/kerneldense3/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_1784709
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ï
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv1a/kernel/Read/ReadVariableOpconv1a/bias/Read/ReadVariableOp!conv1b/kernel/Read/ReadVariableOpconv1b/bias/Read/ReadVariableOp!conv2a/kernel/Read/ReadVariableOpconv2a/bias/Read/ReadVariableOp!conv2b/kernel/Read/ReadVariableOpconv2b/bias/Read/ReadVariableOp!conv2c/kernel/Read/ReadVariableOpconv2c/bias/Read/ReadVariableOp!dense1/kernel/Read/ReadVariableOpdense1/bias/Read/ReadVariableOp!dense2/kernel/Read/ReadVariableOpdense2/bias/Read/ReadVariableOp!dense3/kernel/Read/ReadVariableOpdense3/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_save_1785229

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1a/kernelconv1a/biasconv1b/kernelconv1b/biasconv2a/kernelconv2a/biasconv2b/kernelconv2b/biasconv2c/kernelconv2c/biasdense1/kerneldense1/biasdense2/kerneldense2/biasdense3/kerneldense3/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__traced_restore_1785287Ãõ
§
E
)__inference_flatten_layer_call_fn_1785098

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_17843642
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ*:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
 
_user_specified_nameinputs
Î

Ü
C__inference_conv2c_layer_call_and_return_conditional_losses_1785078

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: **
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:**
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
 ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
 
_user_specified_nameinputs
¿2
¬
D__inference_model_1_layer_call_and_return_conditional_losses_1784501
input_2
conv1a_1784458
conv1a_1784460
conv1b_1784463
conv1b_1784465
conv2a_1784469
conv2a_1784471
conv2b_1784474
conv2b_1784476
conv2c_1784479
conv2c_1784481
dense1_1784485
dense1_1784487
dense2_1784490
dense2_1784492
dense3_1784495
dense3_1784497
identity¢conv1a/StatefulPartitionedCall¢conv1b/StatefulPartitionedCall¢conv2a/StatefulPartitionedCall¢conv2b/StatefulPartitionedCall¢conv2c/StatefulPartitionedCall¢dense1/StatefulPartitionedCall¢dense2/StatefulPartitionedCall¢dense3/StatefulPartitionedCallß
process/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_process_layer_call_and_return_conditional_losses_17842092
process/PartitionedCall²
conv1a/StatefulPartitionedCallStatefulPartitionedCall process/PartitionedCall:output:0conv1a_1784458conv1a_1784460*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5}*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv1a_layer_call_and_return_conditional_losses_17842332 
conv1a/StatefulPartitionedCall¹
conv1b/StatefulPartitionedCallStatefulPartitionedCall'conv1a/StatefulPartitionedCall:output:0conv1b_1784463conv1b_1784465*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3{*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv1b_layer_call_and_return_conditional_losses_17842602 
conv1b/StatefulPartitionedCallû
pool1c/PartitionedCallPartitionedCall'conv1b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_pool1c_layer_call_and_return_conditional_losses_17841612
pool1c/PartitionedCall±
conv2a/StatefulPartitionedCallStatefulPartitionedCallpool1c/PartitionedCall:output:0conv2a_1784469conv2a_1784471*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2a_layer_call_and_return_conditional_losses_17842882 
conv2a/StatefulPartitionedCall¹
conv2b/StatefulPartitionedCallStatefulPartitionedCall'conv2a/StatefulPartitionedCall:output:0conv2b_1784474conv2b_1784476*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2b_layer_call_and_return_conditional_losses_17843152 
conv2b/StatefulPartitionedCall¹
conv2c/StatefulPartitionedCallStatefulPartitionedCall'conv2b/StatefulPartitionedCall:output:0conv2c_1784479conv2c_1784481*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2c_layer_call_and_return_conditional_losses_17843422 
conv2c/StatefulPartitionedCall÷
flatten/PartitionedCallPartitionedCall'conv2c/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_17843642
flatten/PartitionedCall«
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_1784485dense1_1784487*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_17843832 
dense1/StatefulPartitionedCall±
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_1784490dense2_1784492*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense2_layer_call_and_return_conditional_losses_17844102 
dense2/StatefulPartitionedCall±
dense3/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0dense3_1784495dense3_1784497*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense3_layer_call_and_return_conditional_losses_17844372 
dense3/StatefulPartitionedCall
IdentityIdentity'dense3/StatefulPartitionedCall:output:0^conv1a/StatefulPartitionedCall^conv1b/StatefulPartitionedCall^conv2a/StatefulPartitionedCall^conv2b/StatefulPartitionedCall^conv2c/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall^dense3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ À::::::::::::::::2@
conv1a/StatefulPartitionedCallconv1a/StatefulPartitionedCall2@
conv1b/StatefulPartitionedCallconv1b/StatefulPartitionedCall2@
conv2a/StatefulPartitionedCallconv2a/StatefulPartitionedCall2@
conv2b/StatefulPartitionedCallconv2b/StatefulPartitionedCall2@
conv2c/StatefulPartitionedCallconv2c/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2@
dense3/StatefulPartitionedCalldense3/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
!
_user_specified_name	input_2
ö	
Ü
C__inference_dense1_layer_call_and_return_conditional_losses_1785109

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý
}
(__inference_conv2c_layer_call_fn_1785087

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2c_layer_call_and_return_conditional_losses_17843422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
 ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
 
_user_specified_nameinputs
ý
}
(__inference_conv2b_layer_call_fn_1785067

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2b_layer_call_and_return_conditional_losses_17843152
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¼D
ò
#__inference__traced_restore_1785287
file_prefix"
assignvariableop_conv1a_kernel"
assignvariableop_1_conv1a_bias$
 assignvariableop_2_conv1b_kernel"
assignvariableop_3_conv1b_bias$
 assignvariableop_4_conv2a_kernel"
assignvariableop_5_conv2a_bias$
 assignvariableop_6_conv2b_kernel"
assignvariableop_7_conv2b_bias$
 assignvariableop_8_conv2c_kernel"
assignvariableop_9_conv2c_bias%
!assignvariableop_10_dense1_kernel#
assignvariableop_11_dense1_bias%
!assignvariableop_12_dense2_kernel#
assignvariableop_13_dense2_bias%
!assignvariableop_14_dense3_kernel#
assignvariableop_15_dense3_bias
identity_17¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*£
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names°
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*X
_output_shapesF
D:::::::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_conv1a_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1£
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1a_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¥
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv1b_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3£
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv1b_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¥
AssignVariableOp_4AssignVariableOp assignvariableop_4_conv2a_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5£
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv2a_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¥
AssignVariableOp_6AssignVariableOp assignvariableop_6_conv2b_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7£
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv2b_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¥
AssignVariableOp_8AssignVariableOp assignvariableop_8_conv2c_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9£
AssignVariableOp_9AssignVariableOpassignvariableop_9_conv2c_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10©
AssignVariableOp_10AssignVariableOp!assignvariableop_10_dense1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11§
AssignVariableOp_11AssignVariableOpassignvariableop_11_dense1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12©
AssignVariableOp_12AssignVariableOp!assignvariableop_12_dense2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13§
AssignVariableOp_13AssignVariableOpassignvariableop_13_dense2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14©
AssignVariableOp_14AssignVariableOp!assignvariableop_14_dense3_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15§
AssignVariableOp_15AssignVariableOpassignvariableop_15_dense3_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_159
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp¾
Identity_16Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_16±
Identity_17IdentityIdentity_16:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_17"#
identity_17Identity_17:output:0*U
_input_shapesD
B: ::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ã	
Ü
C__inference_dense3_layer_call_and_return_conditional_losses_1785149

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù
_
C__inference_pool1c_layer_call_and_return_conditional_losses_1784161

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 
D
(__inference_pool1c_layer_call_fn_1784167

inputs
identityç
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_pool1c_layer_call_and_return_conditional_losses_17841612
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð

Ü
C__inference_conv1a_layer_call_and_return_conditional_losses_1784233

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5}*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5}2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5}2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5}2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿmý::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý
 
_user_specified_nameinputs
»
E
)__inference_process_layer_call_fn_1784982

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_process_layer_call_and_return_conditional_losses_17841902
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ À:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
 
_user_specified_nameinputs
Î

Ü
C__inference_conv2b_layer_call_and_return_conditional_losses_1784315

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ß
}
(__inference_dense2_layer_call_fn_1785138

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense2_layer_call_and_return_conditional_losses_17844102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ør
ò
"__inference__wrapped_model_1784155
input_21
-model_1_conv1a_conv2d_readvariableop_resource2
.model_1_conv1a_biasadd_readvariableop_resource1
-model_1_conv1b_conv2d_readvariableop_resource2
.model_1_conv1b_biasadd_readvariableop_resource1
-model_1_conv2a_conv2d_readvariableop_resource2
.model_1_conv2a_biasadd_readvariableop_resource1
-model_1_conv2b_conv2d_readvariableop_resource2
.model_1_conv2b_biasadd_readvariableop_resource1
-model_1_conv2c_conv2d_readvariableop_resource2
.model_1_conv2c_biasadd_readvariableop_resource1
-model_1_dense1_matmul_readvariableop_resource2
.model_1_dense1_biasadd_readvariableop_resource1
-model_1_dense2_matmul_readvariableop_resource2
.model_1_dense2_biasadd_readvariableop_resource1
-model_1_dense3_matmul_readvariableop_resource2
.model_1_dense3_biasadd_readvariableop_resource
identity¢%model_1/conv1a/BiasAdd/ReadVariableOp¢$model_1/conv1a/Conv2D/ReadVariableOp¢%model_1/conv1b/BiasAdd/ReadVariableOp¢$model_1/conv1b/Conv2D/ReadVariableOp¢%model_1/conv2a/BiasAdd/ReadVariableOp¢$model_1/conv2a/Conv2D/ReadVariableOp¢%model_1/conv2b/BiasAdd/ReadVariableOp¢$model_1/conv2b/Conv2D/ReadVariableOp¢%model_1/conv2c/BiasAdd/ReadVariableOp¢$model_1/conv2c/Conv2D/ReadVariableOp¢%model_1/dense1/BiasAdd/ReadVariableOp¢$model_1/dense1/MatMul/ReadVariableOp¢%model_1/dense2/BiasAdd/ReadVariableOp¢$model_1/dense2/MatMul/ReadVariableOp¢%model_1/dense3/BiasAdd/ReadVariableOp¢$model_1/dense3/MatMul/ReadVariableOp
#model_1/process/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    8       2%
#model_1/process/strided_slice/stack£
%model_1/process/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"       @  2'
%model_1/process/strided_slice/stack_1£
%model_1/process/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2'
%model_1/process/strided_slice/stack_2Ï
model_1/process/strided_sliceStridedSliceinput_2,model_1/process/strided_slice/stack:output:0.model_1/process/strided_slice/stack_1:output:0.model_1/process/strided_slice/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿXÀ*

begin_mask*
end_mask2
model_1/process/strided_slice{
model_1/process/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
model_1/process/truediv/yÄ
model_1/process/truedivRealDiv&model_1/process/strided_slice:output:0"model_1/process/truediv/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿXÀ2
model_1/process/truedivs
model_1/process/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_1/process/add/y«
model_1/process/addAddV2model_1/process/truediv:z:0model_1/process/add/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿXÀ2
model_1/process/add
model_1/process/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"m   ý   2
model_1/process/resize/sizeô
%model_1/process/resize/ResizeBilinearResizeBilinearmodel_1/process/add:z:0$model_1/process/resize/size:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý*
half_pixel_centers(2'
%model_1/process/resize/ResizeBilinearª
model_1/process/RGBToHSVRGBToHSV6model_1/process/resize/ResizeBilinear:resized_images:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý2
model_1/process/RGBToHSV
%model_1/process/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2'
%model_1/process/strided_slice_1/stack£
'model_1/process/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'model_1/process/strided_slice_1/stack_1£
'model_1/process/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'model_1/process/strided_slice_1/stack_2æ
model_1/process/strided_slice_1StridedSlice!model_1/process/RGBToHSV:output:0.model_1/process/strided_slice_1/stack:output:00model_1/process/strided_slice_1/stack_1:output:00model_1/process/strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý*
ellipsis_mask2!
model_1/process/strided_slice_1Â
$model_1/conv1a/Conv2D/ReadVariableOpReadVariableOp-model_1_conv1a_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$model_1/conv1a/Conv2D/ReadVariableOpó
model_1/conv1a/Conv2DConv2D(model_1/process/strided_slice_1:output:0,model_1/conv1a/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5}*
paddingVALID*
strides
2
model_1/conv1a/Conv2D¹
%model_1/conv1a/BiasAdd/ReadVariableOpReadVariableOp.model_1_conv1a_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model_1/conv1a/BiasAdd/ReadVariableOpÄ
model_1/conv1a/BiasAddBiasAddmodel_1/conv1a/Conv2D:output:0-model_1/conv1a/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5}2
model_1/conv1a/BiasAdd
model_1/conv1a/ReluRelumodel_1/conv1a/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5}2
model_1/conv1a/ReluÂ
$model_1/conv1b/Conv2D/ReadVariableOpReadVariableOp-model_1_conv1b_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$model_1/conv1b/Conv2D/ReadVariableOpì
model_1/conv1b/Conv2DConv2D!model_1/conv1a/Relu:activations:0,model_1/conv1b/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3{*
paddingVALID*
strides
2
model_1/conv1b/Conv2D¹
%model_1/conv1b/BiasAdd/ReadVariableOpReadVariableOp.model_1_conv1b_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model_1/conv1b/BiasAdd/ReadVariableOpÄ
model_1/conv1b/BiasAddBiasAddmodel_1/conv1b/Conv2D:output:0-model_1/conv1b/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3{2
model_1/conv1b/BiasAdd
model_1/conv1b/ReluRelumodel_1/conv1b/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3{2
model_1/conv1b/ReluË
model_1/pool1c/MaxPoolMaxPool!model_1/conv1b/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*
ksize
*
paddingVALID*
strides
2
model_1/pool1c/MaxPoolÂ
$model_1/conv2a/Conv2D/ReadVariableOpReadVariableOp-model_1_conv2a_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02&
$model_1/conv2a/Conv2D/ReadVariableOpê
model_1/conv2a/Conv2DConv2Dmodel_1/pool1c/MaxPool:output:0,model_1/conv2a/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
model_1/conv2a/Conv2D¹
%model_1/conv2a/BiasAdd/ReadVariableOpReadVariableOp.model_1_conv2a_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%model_1/conv2a/BiasAdd/ReadVariableOpÄ
model_1/conv2a/BiasAddBiasAddmodel_1/conv2a/Conv2D:output:0-model_1/conv2a/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
model_1/conv2a/BiasAdd
model_1/conv2a/ReluRelumodel_1/conv2a/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
model_1/conv2a/ReluÂ
$model_1/conv2b/Conv2D/ReadVariableOpReadVariableOp-model_1_conv2b_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02&
$model_1/conv2b/Conv2D/ReadVariableOpì
model_1/conv2b/Conv2DConv2D!model_1/conv2a/Relu:activations:0,model_1/conv2b/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *
paddingVALID*
strides
2
model_1/conv2b/Conv2D¹
%model_1/conv2b/BiasAdd/ReadVariableOpReadVariableOp.model_1_conv2b_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%model_1/conv2b/BiasAdd/ReadVariableOpÄ
model_1/conv2b/BiasAddBiasAddmodel_1/conv2b/Conv2D:output:0-model_1/conv2b/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
model_1/conv2b/BiasAdd
model_1/conv2b/ReluRelumodel_1/conv2b/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
model_1/conv2b/ReluÂ
$model_1/conv2c/Conv2D/ReadVariableOpReadVariableOp-model_1_conv2c_conv2d_readvariableop_resource*&
_output_shapes
: **
dtype02&
$model_1/conv2c/Conv2D/ReadVariableOpì
model_1/conv2c/Conv2DConv2D!model_1/conv2b/Relu:activations:0,model_1/conv2c/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
paddingVALID*
strides
2
model_1/conv2c/Conv2D¹
%model_1/conv2c/BiasAdd/ReadVariableOpReadVariableOp.model_1_conv2c_biasadd_readvariableop_resource*
_output_shapes
:**
dtype02'
%model_1/conv2c/BiasAdd/ReadVariableOpÄ
model_1/conv2c/BiasAddBiasAddmodel_1/conv2c/Conv2D:output:0-model_1/conv2c/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
model_1/conv2c/BiasAdd
model_1/conv2c/ReluRelumodel_1/conv2c/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
model_1/conv2c/Relu
model_1/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
model_1/flatten/Const³
model_1/flatten/ReshapeReshape!model_1/conv2c/Relu:activations:0model_1/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/flatten/Reshape¼
$model_1/dense1/MatMul/ReadVariableOpReadVariableOp-model_1_dense1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02&
$model_1/dense1/MatMul/ReadVariableOp»
model_1/dense1/MatMulMatMul model_1/flatten/Reshape:output:0,model_1/dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense1/MatMulº
%model_1/dense1/BiasAdd/ReadVariableOpReadVariableOp.model_1_dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02'
%model_1/dense1/BiasAdd/ReadVariableOp¾
model_1/dense1/BiasAddBiasAddmodel_1/dense1/MatMul:product:0-model_1/dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense1/BiasAdd
model_1/dense1/ReluRelumodel_1/dense1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense1/Relu»
$model_1/dense2/MatMul/ReadVariableOpReadVariableOp-model_1_dense2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02&
$model_1/dense2/MatMul/ReadVariableOp»
model_1/dense2/MatMulMatMul!model_1/dense1/Relu:activations:0,model_1/dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense2/MatMul¹
%model_1/dense2/BiasAdd/ReadVariableOpReadVariableOp.model_1_dense2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model_1/dense2/BiasAdd/ReadVariableOp½
model_1/dense2/BiasAddBiasAddmodel_1/dense2/MatMul:product:0-model_1/dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense2/BiasAdd
model_1/dense2/ReluRelumodel_1/dense2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense2/Reluº
$model_1/dense3/MatMul/ReadVariableOpReadVariableOp-model_1_dense3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02&
$model_1/dense3/MatMul/ReadVariableOp»
model_1/dense3/MatMulMatMul!model_1/dense2/Relu:activations:0,model_1/dense3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense3/MatMul¹
%model_1/dense3/BiasAdd/ReadVariableOpReadVariableOp.model_1_dense3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model_1/dense3/BiasAdd/ReadVariableOp½
model_1/dense3/BiasAddBiasAddmodel_1/dense3/MatMul:product:0-model_1/dense3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense3/BiasAdd
model_1/dense3/TanhTanhmodel_1/dense3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
model_1/dense3/Tanhã
IdentityIdentitymodel_1/dense3/Tanh:y:0&^model_1/conv1a/BiasAdd/ReadVariableOp%^model_1/conv1a/Conv2D/ReadVariableOp&^model_1/conv1b/BiasAdd/ReadVariableOp%^model_1/conv1b/Conv2D/ReadVariableOp&^model_1/conv2a/BiasAdd/ReadVariableOp%^model_1/conv2a/Conv2D/ReadVariableOp&^model_1/conv2b/BiasAdd/ReadVariableOp%^model_1/conv2b/Conv2D/ReadVariableOp&^model_1/conv2c/BiasAdd/ReadVariableOp%^model_1/conv2c/Conv2D/ReadVariableOp&^model_1/dense1/BiasAdd/ReadVariableOp%^model_1/dense1/MatMul/ReadVariableOp&^model_1/dense2/BiasAdd/ReadVariableOp%^model_1/dense2/MatMul/ReadVariableOp&^model_1/dense3/BiasAdd/ReadVariableOp%^model_1/dense3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ À::::::::::::::::2N
%model_1/conv1a/BiasAdd/ReadVariableOp%model_1/conv1a/BiasAdd/ReadVariableOp2L
$model_1/conv1a/Conv2D/ReadVariableOp$model_1/conv1a/Conv2D/ReadVariableOp2N
%model_1/conv1b/BiasAdd/ReadVariableOp%model_1/conv1b/BiasAdd/ReadVariableOp2L
$model_1/conv1b/Conv2D/ReadVariableOp$model_1/conv1b/Conv2D/ReadVariableOp2N
%model_1/conv2a/BiasAdd/ReadVariableOp%model_1/conv2a/BiasAdd/ReadVariableOp2L
$model_1/conv2a/Conv2D/ReadVariableOp$model_1/conv2a/Conv2D/ReadVariableOp2N
%model_1/conv2b/BiasAdd/ReadVariableOp%model_1/conv2b/BiasAdd/ReadVariableOp2L
$model_1/conv2b/Conv2D/ReadVariableOp$model_1/conv2b/Conv2D/ReadVariableOp2N
%model_1/conv2c/BiasAdd/ReadVariableOp%model_1/conv2c/BiasAdd/ReadVariableOp2L
$model_1/conv2c/Conv2D/ReadVariableOp$model_1/conv2c/Conv2D/ReadVariableOp2N
%model_1/dense1/BiasAdd/ReadVariableOp%model_1/dense1/BiasAdd/ReadVariableOp2L
$model_1/dense1/MatMul/ReadVariableOp$model_1/dense1/MatMul/ReadVariableOp2N
%model_1/dense2/BiasAdd/ReadVariableOp%model_1/dense2/BiasAdd/ReadVariableOp2L
$model_1/dense2/MatMul/ReadVariableOp$model_1/dense2/MatMul/ReadVariableOp2N
%model_1/dense3/BiasAdd/ReadVariableOp%model_1/dense3/BiasAdd/ReadVariableOp2L
$model_1/dense3/MatMul/ReadVariableOp$model_1/dense3/MatMul/ReadVariableOp:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
!
_user_specified_name	input_2
þa


D__inference_model_1_layer_call_and_return_conditional_losses_1784787

inputs)
%conv1a_conv2d_readvariableop_resource*
&conv1a_biasadd_readvariableop_resource)
%conv1b_conv2d_readvariableop_resource*
&conv1b_biasadd_readvariableop_resource)
%conv2a_conv2d_readvariableop_resource*
&conv2a_biasadd_readvariableop_resource)
%conv2b_conv2d_readvariableop_resource*
&conv2b_biasadd_readvariableop_resource)
%conv2c_conv2d_readvariableop_resource*
&conv2c_biasadd_readvariableop_resource)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%dense2_matmul_readvariableop_resource*
&dense2_biasadd_readvariableop_resource)
%dense3_matmul_readvariableop_resource*
&dense3_biasadd_readvariableop_resource
identity¢conv1a/BiasAdd/ReadVariableOp¢conv1a/Conv2D/ReadVariableOp¢conv1b/BiasAdd/ReadVariableOp¢conv1b/Conv2D/ReadVariableOp¢conv2a/BiasAdd/ReadVariableOp¢conv2a/Conv2D/ReadVariableOp¢conv2b/BiasAdd/ReadVariableOp¢conv2b/Conv2D/ReadVariableOp¢conv2c/BiasAdd/ReadVariableOp¢conv2c/Conv2D/ReadVariableOp¢dense1/BiasAdd/ReadVariableOp¢dense1/MatMul/ReadVariableOp¢dense2/BiasAdd/ReadVariableOp¢dense2/MatMul/ReadVariableOp¢dense3/BiasAdd/ReadVariableOp¢dense3/MatMul/ReadVariableOp
process/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    8       2
process/strided_slice/stack
process/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"       @  2
process/strided_slice/stack_1
process/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
process/strided_slice/stack_2¦
process/strided_sliceStridedSliceinputs$process/strided_slice/stack:output:0&process/strided_slice/stack_1:output:0&process/strided_slice/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿXÀ*

begin_mask*
end_mask2
process/strided_slicek
process/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
process/truediv/y¤
process/truedivRealDivprocess/strided_slice:output:0process/truediv/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿXÀ2
process/truedivc
process/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
process/add/y
process/addAddV2process/truediv:z:0process/add/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿXÀ2
process/add{
process/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"m   ý   2
process/resize/sizeÔ
process/resize/ResizeBilinearResizeBilinearprocess/add:z:0process/resize/size:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý*
half_pixel_centers(2
process/resize/ResizeBilinear
process/RGBToHSVRGBToHSV.process/resize/ResizeBilinear:resized_images:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý2
process/RGBToHSV
process/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
process/strided_slice_1/stack
process/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
process/strided_slice_1/stack_1
process/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
process/strided_slice_1/stack_2¶
process/strided_slice_1StridedSliceprocess/RGBToHSV:output:0&process/strided_slice_1/stack:output:0(process/strided_slice_1/stack_1:output:0(process/strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý*
ellipsis_mask2
process/strided_slice_1ª
conv1a/Conv2D/ReadVariableOpReadVariableOp%conv1a_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1a/Conv2D/ReadVariableOpÓ
conv1a/Conv2DConv2D process/strided_slice_1:output:0$conv1a/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5}*
paddingVALID*
strides
2
conv1a/Conv2D¡
conv1a/BiasAdd/ReadVariableOpReadVariableOp&conv1a_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1a/BiasAdd/ReadVariableOp¤
conv1a/BiasAddBiasAddconv1a/Conv2D:output:0%conv1a/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5}2
conv1a/BiasAddu
conv1a/ReluReluconv1a/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5}2
conv1a/Reluª
conv1b/Conv2D/ReadVariableOpReadVariableOp%conv1b_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1b/Conv2D/ReadVariableOpÌ
conv1b/Conv2DConv2Dconv1a/Relu:activations:0$conv1b/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3{*
paddingVALID*
strides
2
conv1b/Conv2D¡
conv1b/BiasAdd/ReadVariableOpReadVariableOp&conv1b_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1b/BiasAdd/ReadVariableOp¤
conv1b/BiasAddBiasAddconv1b/Conv2D:output:0%conv1b/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3{2
conv1b/BiasAddu
conv1b/ReluReluconv1b/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3{2
conv1b/Relu³
pool1c/MaxPoolMaxPoolconv1b/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*
ksize
*
paddingVALID*
strides
2
pool1c/MaxPoolª
conv2a/Conv2D/ReadVariableOpReadVariableOp%conv2a_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2a/Conv2D/ReadVariableOpÊ
conv2a/Conv2DConv2Dpool1c/MaxPool:output:0$conv2a/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2a/Conv2D¡
conv2a/BiasAdd/ReadVariableOpReadVariableOp&conv2a_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2a/BiasAdd/ReadVariableOp¤
conv2a/BiasAddBiasAddconv2a/Conv2D:output:0%conv2a/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2a/BiasAddu
conv2a/ReluReluconv2a/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2a/Reluª
conv2b/Conv2D/ReadVariableOpReadVariableOp%conv2b_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
conv2b/Conv2D/ReadVariableOpÌ
conv2b/Conv2DConv2Dconv2a/Relu:activations:0$conv2b/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *
paddingVALID*
strides
2
conv2b/Conv2D¡
conv2b/BiasAdd/ReadVariableOpReadVariableOp&conv2b_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2b/BiasAdd/ReadVariableOp¤
conv2b/BiasAddBiasAddconv2b/Conv2D:output:0%conv2b/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
conv2b/BiasAddu
conv2b/ReluReluconv2b/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
conv2b/Reluª
conv2c/Conv2D/ReadVariableOpReadVariableOp%conv2c_conv2d_readvariableop_resource*&
_output_shapes
: **
dtype02
conv2c/Conv2D/ReadVariableOpÌ
conv2c/Conv2DConv2Dconv2b/Relu:activations:0$conv2c/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
paddingVALID*
strides
2
conv2c/Conv2D¡
conv2c/BiasAdd/ReadVariableOpReadVariableOp&conv2c_biasadd_readvariableop_resource*
_output_shapes
:**
dtype02
conv2c/BiasAdd/ReadVariableOp¤
conv2c/BiasAddBiasAddconv2c/Conv2D:output:0%conv2c/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
conv2c/BiasAddu
conv2c/ReluReluconv2c/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
conv2c/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
flatten/Const
flatten/ReshapeReshapeconv2c/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten/Reshape¤
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense1/MatMul/ReadVariableOp
dense1/MatMulMatMulflatten/Reshape:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense1/MatMul¢
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense1/BiasAdd/ReadVariableOp
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense1/BiasAddn
dense1/ReluReludense1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense1/Relu£
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense2/MatMul/ReadVariableOp
dense2/MatMulMatMuldense1/Relu:activations:0$dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense2/MatMul¡
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense2/BiasAdd/ReadVariableOp
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense2/BiasAddm
dense2/ReluReludense2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense2/Relu¢
dense3/MatMul/ReadVariableOpReadVariableOp%dense3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense3/MatMul/ReadVariableOp
dense3/MatMulMatMuldense2/Relu:activations:0$dense3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense3/MatMul¡
dense3/BiasAdd/ReadVariableOpReadVariableOp&dense3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense3/BiasAdd/ReadVariableOp
dense3/BiasAddBiasAdddense3/MatMul:product:0%dense3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense3/BiasAddm
dense3/TanhTanhdense3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense3/TanhÛ
IdentityIdentitydense3/Tanh:y:0^conv1a/BiasAdd/ReadVariableOp^conv1a/Conv2D/ReadVariableOp^conv1b/BiasAdd/ReadVariableOp^conv1b/Conv2D/ReadVariableOp^conv2a/BiasAdd/ReadVariableOp^conv2a/Conv2D/ReadVariableOp^conv2b/BiasAdd/ReadVariableOp^conv2b/Conv2D/ReadVariableOp^conv2c/BiasAdd/ReadVariableOp^conv2c/Conv2D/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp^dense2/BiasAdd/ReadVariableOp^dense2/MatMul/ReadVariableOp^dense3/BiasAdd/ReadVariableOp^dense3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ À::::::::::::::::2>
conv1a/BiasAdd/ReadVariableOpconv1a/BiasAdd/ReadVariableOp2<
conv1a/Conv2D/ReadVariableOpconv1a/Conv2D/ReadVariableOp2>
conv1b/BiasAdd/ReadVariableOpconv1b/BiasAdd/ReadVariableOp2<
conv1b/Conv2D/ReadVariableOpconv1b/Conv2D/ReadVariableOp2>
conv2a/BiasAdd/ReadVariableOpconv2a/BiasAdd/ReadVariableOp2<
conv2a/Conv2D/ReadVariableOpconv2a/Conv2D/ReadVariableOp2>
conv2b/BiasAdd/ReadVariableOpconv2b/BiasAdd/ReadVariableOp2<
conv2b/Conv2D/ReadVariableOpconv2b/Conv2D/ReadVariableOp2>
conv2c/BiasAdd/ReadVariableOpconv2c/BiasAdd/ReadVariableOp2<
conv2c/Conv2D/ReadVariableOpconv2c/Conv2D/ReadVariableOp2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp2>
dense2/BiasAdd/ReadVariableOpdense2/BiasAdd/ReadVariableOp2<
dense2/MatMul/ReadVariableOpdense2/MatMul/ReadVariableOp2>
dense3/BiasAdd/ReadVariableOpdense3/BiasAdd/ReadVariableOp2<
dense3/MatMul/ReadVariableOpdense3/MatMul/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
 
_user_specified_nameinputs
ð	
Ü
C__inference_dense2_layer_call_and_return_conditional_losses_1784410

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î

Ü
C__inference_conv2a_layer_call_and_return_conditional_losses_1784288

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ=::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
¼2
«
D__inference_model_1_layer_call_and_return_conditional_losses_1784551

inputs
conv1a_1784508
conv1a_1784510
conv1b_1784513
conv1b_1784515
conv2a_1784519
conv2a_1784521
conv2b_1784524
conv2b_1784526
conv2c_1784529
conv2c_1784531
dense1_1784535
dense1_1784537
dense2_1784540
dense2_1784542
dense3_1784545
dense3_1784547
identity¢conv1a/StatefulPartitionedCall¢conv1b/StatefulPartitionedCall¢conv2a/StatefulPartitionedCall¢conv2b/StatefulPartitionedCall¢conv2c/StatefulPartitionedCall¢dense1/StatefulPartitionedCall¢dense2/StatefulPartitionedCall¢dense3/StatefulPartitionedCallÞ
process/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_process_layer_call_and_return_conditional_losses_17841902
process/PartitionedCall²
conv1a/StatefulPartitionedCallStatefulPartitionedCall process/PartitionedCall:output:0conv1a_1784508conv1a_1784510*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5}*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv1a_layer_call_and_return_conditional_losses_17842332 
conv1a/StatefulPartitionedCall¹
conv1b/StatefulPartitionedCallStatefulPartitionedCall'conv1a/StatefulPartitionedCall:output:0conv1b_1784513conv1b_1784515*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3{*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv1b_layer_call_and_return_conditional_losses_17842602 
conv1b/StatefulPartitionedCallû
pool1c/PartitionedCallPartitionedCall'conv1b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_pool1c_layer_call_and_return_conditional_losses_17841612
pool1c/PartitionedCall±
conv2a/StatefulPartitionedCallStatefulPartitionedCallpool1c/PartitionedCall:output:0conv2a_1784519conv2a_1784521*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2a_layer_call_and_return_conditional_losses_17842882 
conv2a/StatefulPartitionedCall¹
conv2b/StatefulPartitionedCallStatefulPartitionedCall'conv2a/StatefulPartitionedCall:output:0conv2b_1784524conv2b_1784526*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2b_layer_call_and_return_conditional_losses_17843152 
conv2b/StatefulPartitionedCall¹
conv2c/StatefulPartitionedCallStatefulPartitionedCall'conv2b/StatefulPartitionedCall:output:0conv2c_1784529conv2c_1784531*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2c_layer_call_and_return_conditional_losses_17843422 
conv2c/StatefulPartitionedCall÷
flatten/PartitionedCallPartitionedCall'conv2c/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_17843642
flatten/PartitionedCall«
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_1784535dense1_1784537*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_17843832 
dense1/StatefulPartitionedCall±
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_1784540dense2_1784542*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense2_layer_call_and_return_conditional_losses_17844102 
dense2/StatefulPartitionedCall±
dense3/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0dense3_1784545dense3_1784547*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense3_layer_call_and_return_conditional_losses_17844372 
dense3/StatefulPartitionedCall
IdentityIdentity'dense3/StatefulPartitionedCall:output:0^conv1a/StatefulPartitionedCall^conv1b/StatefulPartitionedCall^conv2a/StatefulPartitionedCall^conv2b/StatefulPartitionedCall^conv2c/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall^dense3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ À::::::::::::::::2@
conv1a/StatefulPartitionedCallconv1a/StatefulPartitionedCall2@
conv1b/StatefulPartitionedCallconv1b/StatefulPartitionedCall2@
conv2a/StatefulPartitionedCallconv2a/StatefulPartitionedCall2@
conv2b/StatefulPartitionedCallconv2b/StatefulPartitionedCall2@
conv2c/StatefulPartitionedCallconv2c/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2@
dense3/StatefulPartitionedCalldense3/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
 
_user_specified_nameinputs
Î

Ü
C__inference_conv1b_layer_call_and_return_conditional_losses_1784260

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3{*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3{2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3{2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3{2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ5}::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5}
 
_user_specified_nameinputs
Á

Ö
)__inference_model_1_layer_call_fn_1784670
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity¢StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_17846352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ À::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
!
_user_specified_name	input_2
¼
`
D__inference_flatten_layer_call_and_return_conditional_losses_1785093

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ*:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
 
_user_specified_nameinputs
Î

Ü
C__inference_conv2c_layer_call_and_return_conditional_losses_1784342

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: **
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:**
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ
 ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
 
_user_specified_nameinputs
¼2
«
D__inference_model_1_layer_call_and_return_conditional_losses_1784635

inputs
conv1a_1784592
conv1a_1784594
conv1b_1784597
conv1b_1784599
conv2a_1784603
conv2a_1784605
conv2b_1784608
conv2b_1784610
conv2c_1784613
conv2c_1784615
dense1_1784619
dense1_1784621
dense2_1784624
dense2_1784626
dense3_1784629
dense3_1784631
identity¢conv1a/StatefulPartitionedCall¢conv1b/StatefulPartitionedCall¢conv2a/StatefulPartitionedCall¢conv2b/StatefulPartitionedCall¢conv2c/StatefulPartitionedCall¢dense1/StatefulPartitionedCall¢dense2/StatefulPartitionedCall¢dense3/StatefulPartitionedCallÞ
process/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_process_layer_call_and_return_conditional_losses_17842092
process/PartitionedCall²
conv1a/StatefulPartitionedCallStatefulPartitionedCall process/PartitionedCall:output:0conv1a_1784592conv1a_1784594*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5}*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv1a_layer_call_and_return_conditional_losses_17842332 
conv1a/StatefulPartitionedCall¹
conv1b/StatefulPartitionedCallStatefulPartitionedCall'conv1a/StatefulPartitionedCall:output:0conv1b_1784597conv1b_1784599*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3{*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv1b_layer_call_and_return_conditional_losses_17842602 
conv1b/StatefulPartitionedCallû
pool1c/PartitionedCallPartitionedCall'conv1b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_pool1c_layer_call_and_return_conditional_losses_17841612
pool1c/PartitionedCall±
conv2a/StatefulPartitionedCallStatefulPartitionedCallpool1c/PartitionedCall:output:0conv2a_1784603conv2a_1784605*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2a_layer_call_and_return_conditional_losses_17842882 
conv2a/StatefulPartitionedCall¹
conv2b/StatefulPartitionedCallStatefulPartitionedCall'conv2a/StatefulPartitionedCall:output:0conv2b_1784608conv2b_1784610*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2b_layer_call_and_return_conditional_losses_17843152 
conv2b/StatefulPartitionedCall¹
conv2c/StatefulPartitionedCallStatefulPartitionedCall'conv2b/StatefulPartitionedCall:output:0conv2c_1784613conv2c_1784615*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2c_layer_call_and_return_conditional_losses_17843422 
conv2c/StatefulPartitionedCall÷
flatten/PartitionedCallPartitionedCall'conv2c/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_17843642
flatten/PartitionedCall«
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_1784619dense1_1784621*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_17843832 
dense1/StatefulPartitionedCall±
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_1784624dense2_1784626*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense2_layer_call_and_return_conditional_losses_17844102 
dense2/StatefulPartitionedCall±
dense3/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0dense3_1784629dense3_1784631*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense3_layer_call_and_return_conditional_losses_17844372 
dense3/StatefulPartitionedCall
IdentityIdentity'dense3/StatefulPartitionedCall:output:0^conv1a/StatefulPartitionedCall^conv1b/StatefulPartitionedCall^conv2a/StatefulPartitionedCall^conv2b/StatefulPartitionedCall^conv2c/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall^dense3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ À::::::::::::::::2@
conv1a/StatefulPartitionedCallconv1a/StatefulPartitionedCall2@
conv1b/StatefulPartitionedCallconv1b/StatefulPartitionedCall2@
conv2a/StatefulPartitionedCallconv2a/StatefulPartitionedCall2@
conv2b/StatefulPartitionedCallconv2b/StatefulPartitionedCall2@
conv2c/StatefulPartitionedCallconv2c/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2@
dense3/StatefulPartitionedCalldense3/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
 
_user_specified_nameinputs
Á

Ö
)__inference_model_1_layer_call_fn_1784586
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity¢StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_17845512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ À::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
!
_user_specified_name	input_2
¿2
¬
D__inference_model_1_layer_call_and_return_conditional_losses_1784454
input_2
conv1a_1784244
conv1a_1784246
conv1b_1784271
conv1b_1784273
conv2a_1784299
conv2a_1784301
conv2b_1784326
conv2b_1784328
conv2c_1784353
conv2c_1784355
dense1_1784394
dense1_1784396
dense2_1784421
dense2_1784423
dense3_1784448
dense3_1784450
identity¢conv1a/StatefulPartitionedCall¢conv1b/StatefulPartitionedCall¢conv2a/StatefulPartitionedCall¢conv2b/StatefulPartitionedCall¢conv2c/StatefulPartitionedCall¢dense1/StatefulPartitionedCall¢dense2/StatefulPartitionedCall¢dense3/StatefulPartitionedCallß
process/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_process_layer_call_and_return_conditional_losses_17841902
process/PartitionedCall²
conv1a/StatefulPartitionedCallStatefulPartitionedCall process/PartitionedCall:output:0conv1a_1784244conv1a_1784246*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5}*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv1a_layer_call_and_return_conditional_losses_17842332 
conv1a/StatefulPartitionedCall¹
conv1b/StatefulPartitionedCallStatefulPartitionedCall'conv1a/StatefulPartitionedCall:output:0conv1b_1784271conv1b_1784273*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3{*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv1b_layer_call_and_return_conditional_losses_17842602 
conv1b/StatefulPartitionedCallû
pool1c/PartitionedCallPartitionedCall'conv1b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_pool1c_layer_call_and_return_conditional_losses_17841612
pool1c/PartitionedCall±
conv2a/StatefulPartitionedCallStatefulPartitionedCallpool1c/PartitionedCall:output:0conv2a_1784299conv2a_1784301*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2a_layer_call_and_return_conditional_losses_17842882 
conv2a/StatefulPartitionedCall¹
conv2b/StatefulPartitionedCallStatefulPartitionedCall'conv2a/StatefulPartitionedCall:output:0conv2b_1784326conv2b_1784328*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2b_layer_call_and_return_conditional_losses_17843152 
conv2b/StatefulPartitionedCall¹
conv2c/StatefulPartitionedCallStatefulPartitionedCall'conv2b/StatefulPartitionedCall:output:0conv2c_1784353conv2c_1784355*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2c_layer_call_and_return_conditional_losses_17843422 
conv2c/StatefulPartitionedCall÷
flatten/PartitionedCallPartitionedCall'conv2c/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_17843642
flatten/PartitionedCall«
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_1784394dense1_1784396*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_17843832 
dense1/StatefulPartitionedCall±
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_1784421dense2_1784423*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense2_layer_call_and_return_conditional_losses_17844102 
dense2/StatefulPartitionedCall±
dense3/StatefulPartitionedCallStatefulPartitionedCall'dense2/StatefulPartitionedCall:output:0dense3_1784448dense3_1784450*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense3_layer_call_and_return_conditional_losses_17844372 
dense3/StatefulPartitionedCall
IdentityIdentity'dense3/StatefulPartitionedCall:output:0^conv1a/StatefulPartitionedCall^conv1b/StatefulPartitionedCall^conv2a/StatefulPartitionedCall^conv2b/StatefulPartitionedCall^conv2c/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall^dense3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ À::::::::::::::::2@
conv1a/StatefulPartitionedCallconv1a/StatefulPartitionedCall2@
conv1b/StatefulPartitionedCallconv1b/StatefulPartitionedCall2@
conv2a/StatefulPartitionedCallconv2a/StatefulPartitionedCall2@
conv2b/StatefulPartitionedCallconv2b/StatefulPartitionedCall2@
conv2c/StatefulPartitionedCallconv2c/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2@
dense3/StatefulPartitionedCalldense3/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
!
_user_specified_name	input_2
ý
}
(__inference_conv2a_layer_call_fn_1785047

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2a_layer_call_and_return_conditional_losses_17842882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ=::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
ð	
Ü
C__inference_dense2_layer_call_and_return_conditional_losses_1785129

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã	
Ü
C__inference_dense3_layer_call_and_return_conditional_losses_1784437

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý
}
(__inference_dense3_layer_call_fn_1785158

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense3_layer_call_and_return_conditional_losses_17844372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼
`
D__inference_flatten_layer_call_and_return_conditional_losses_1784364

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ*:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
 
_user_specified_nameinputs
ï
`
D__inference_process_layer_call_and_return_conditional_losses_1784958

inputs
identity
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    8       2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"       @  2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2þ
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿXÀ*

begin_mask*
end_mask2
strided_slice[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
	truediv/y
truedivRealDivstrided_slice:output:0truediv/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿXÀ2	
truedivS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
add/yk
addAddV2truediv:z:0add/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿXÀ2
addk
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"m   ý   2
resize/size´
resize/ResizeBilinearResizeBilinearadd:z:0resize/size:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý*
half_pixel_centers(2
resize/ResizeBilinearz
RGBToHSVRGBToHSV&resize/ResizeBilinear:resized_images:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý2

RGBToHSV
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceRGBToHSV:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý*
ellipsis_mask2
strided_slice_1u
IdentityIdentitystrided_slice_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ À:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
 
_user_specified_nameinputs
¾

Õ
)__inference_model_1_layer_call_fn_1784939

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity¢StatefulPartitionedCall²
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_17846352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ À::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
 
_user_specified_nameinputs
ÿ
}
(__inference_conv1a_layer_call_fn_1785007

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5}*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv1a_layer_call_and_return_conditional_losses_17842332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5}2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿmý::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý
 
_user_specified_nameinputs
á
}
(__inference_dense1_layer_call_fn_1785118

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_17843832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


Ò
%__inference_signature_wrapper_1784709
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_17841552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ À::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
!
_user_specified_name	input_2
ö	
Ü
C__inference_dense1_layer_call_and_return_conditional_losses_1784383

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þa


D__inference_model_1_layer_call_and_return_conditional_losses_1784865

inputs)
%conv1a_conv2d_readvariableop_resource*
&conv1a_biasadd_readvariableop_resource)
%conv1b_conv2d_readvariableop_resource*
&conv1b_biasadd_readvariableop_resource)
%conv2a_conv2d_readvariableop_resource*
&conv2a_biasadd_readvariableop_resource)
%conv2b_conv2d_readvariableop_resource*
&conv2b_biasadd_readvariableop_resource)
%conv2c_conv2d_readvariableop_resource*
&conv2c_biasadd_readvariableop_resource)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%dense2_matmul_readvariableop_resource*
&dense2_biasadd_readvariableop_resource)
%dense3_matmul_readvariableop_resource*
&dense3_biasadd_readvariableop_resource
identity¢conv1a/BiasAdd/ReadVariableOp¢conv1a/Conv2D/ReadVariableOp¢conv1b/BiasAdd/ReadVariableOp¢conv1b/Conv2D/ReadVariableOp¢conv2a/BiasAdd/ReadVariableOp¢conv2a/Conv2D/ReadVariableOp¢conv2b/BiasAdd/ReadVariableOp¢conv2b/Conv2D/ReadVariableOp¢conv2c/BiasAdd/ReadVariableOp¢conv2c/Conv2D/ReadVariableOp¢dense1/BiasAdd/ReadVariableOp¢dense1/MatMul/ReadVariableOp¢dense2/BiasAdd/ReadVariableOp¢dense2/MatMul/ReadVariableOp¢dense3/BiasAdd/ReadVariableOp¢dense3/MatMul/ReadVariableOp
process/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    8       2
process/strided_slice/stack
process/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"       @  2
process/strided_slice/stack_1
process/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
process/strided_slice/stack_2¦
process/strided_sliceStridedSliceinputs$process/strided_slice/stack:output:0&process/strided_slice/stack_1:output:0&process/strided_slice/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿXÀ*

begin_mask*
end_mask2
process/strided_slicek
process/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
process/truediv/y¤
process/truedivRealDivprocess/strided_slice:output:0process/truediv/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿXÀ2
process/truedivc
process/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
process/add/y
process/addAddV2process/truediv:z:0process/add/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿXÀ2
process/add{
process/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"m   ý   2
process/resize/sizeÔ
process/resize/ResizeBilinearResizeBilinearprocess/add:z:0process/resize/size:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý*
half_pixel_centers(2
process/resize/ResizeBilinear
process/RGBToHSVRGBToHSV.process/resize/ResizeBilinear:resized_images:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý2
process/RGBToHSV
process/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
process/strided_slice_1/stack
process/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
process/strided_slice_1/stack_1
process/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
process/strided_slice_1/stack_2¶
process/strided_slice_1StridedSliceprocess/RGBToHSV:output:0&process/strided_slice_1/stack:output:0(process/strided_slice_1/stack_1:output:0(process/strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý*
ellipsis_mask2
process/strided_slice_1ª
conv1a/Conv2D/ReadVariableOpReadVariableOp%conv1a_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1a/Conv2D/ReadVariableOpÓ
conv1a/Conv2DConv2D process/strided_slice_1:output:0$conv1a/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5}*
paddingVALID*
strides
2
conv1a/Conv2D¡
conv1a/BiasAdd/ReadVariableOpReadVariableOp&conv1a_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1a/BiasAdd/ReadVariableOp¤
conv1a/BiasAddBiasAddconv1a/Conv2D:output:0%conv1a/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5}2
conv1a/BiasAddu
conv1a/ReluReluconv1a/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5}2
conv1a/Reluª
conv1b/Conv2D/ReadVariableOpReadVariableOp%conv1b_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv1b/Conv2D/ReadVariableOpÌ
conv1b/Conv2DConv2Dconv1a/Relu:activations:0$conv1b/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3{*
paddingVALID*
strides
2
conv1b/Conv2D¡
conv1b/BiasAdd/ReadVariableOpReadVariableOp&conv1b_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1b/BiasAdd/ReadVariableOp¤
conv1b/BiasAddBiasAddconv1b/Conv2D:output:0%conv1b/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3{2
conv1b/BiasAddu
conv1b/ReluReluconv1b/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3{2
conv1b/Relu³
pool1c/MaxPoolMaxPoolconv1b/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=*
ksize
*
paddingVALID*
strides
2
pool1c/MaxPoolª
conv2a/Conv2D/ReadVariableOpReadVariableOp%conv2a_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2a/Conv2D/ReadVariableOpÊ
conv2a/Conv2DConv2Dpool1c/MaxPool:output:0$conv2a/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
conv2a/Conv2D¡
conv2a/BiasAdd/ReadVariableOpReadVariableOp&conv2a_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2a/BiasAdd/ReadVariableOp¤
conv2a/BiasAddBiasAddconv2a/Conv2D:output:0%conv2a/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2a/BiasAddu
conv2a/ReluReluconv2a/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2a/Reluª
conv2b/Conv2D/ReadVariableOpReadVariableOp%conv2b_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
conv2b/Conv2D/ReadVariableOpÌ
conv2b/Conv2DConv2Dconv2a/Relu:activations:0$conv2b/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *
paddingVALID*
strides
2
conv2b/Conv2D¡
conv2b/BiasAdd/ReadVariableOpReadVariableOp&conv2b_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2b/BiasAdd/ReadVariableOp¤
conv2b/BiasAddBiasAddconv2b/Conv2D:output:0%conv2b/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
conv2b/BiasAddu
conv2b/ReluReluconv2b/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
conv2b/Reluª
conv2c/Conv2D/ReadVariableOpReadVariableOp%conv2c_conv2d_readvariableop_resource*&
_output_shapes
: **
dtype02
conv2c/Conv2D/ReadVariableOpÌ
conv2c/Conv2DConv2Dconv2b/Relu:activations:0$conv2c/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
paddingVALID*
strides
2
conv2c/Conv2D¡
conv2c/BiasAdd/ReadVariableOpReadVariableOp&conv2c_biasadd_readvariableop_resource*
_output_shapes
:**
dtype02
conv2c/BiasAdd/ReadVariableOp¤
conv2c/BiasAddBiasAddconv2c/Conv2D:output:0%conv2c/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
conv2c/BiasAddu
conv2c/ReluReluconv2c/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
conv2c/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
flatten/Const
flatten/ReshapeReshapeconv2c/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten/Reshape¤
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense1/MatMul/ReadVariableOp
dense1/MatMulMatMulflatten/Reshape:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense1/MatMul¢
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense1/BiasAdd/ReadVariableOp
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense1/BiasAddn
dense1/ReluReludense1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense1/Relu£
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense2/MatMul/ReadVariableOp
dense2/MatMulMatMuldense1/Relu:activations:0$dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense2/MatMul¡
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense2/BiasAdd/ReadVariableOp
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense2/BiasAddm
dense2/ReluReludense2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense2/Relu¢
dense3/MatMul/ReadVariableOpReadVariableOp%dense3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense3/MatMul/ReadVariableOp
dense3/MatMulMatMuldense2/Relu:activations:0$dense3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense3/MatMul¡
dense3/BiasAdd/ReadVariableOpReadVariableOp&dense3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense3/BiasAdd/ReadVariableOp
dense3/BiasAddBiasAdddense3/MatMul:product:0%dense3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense3/BiasAddm
dense3/TanhTanhdense3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense3/TanhÛ
IdentityIdentitydense3/Tanh:y:0^conv1a/BiasAdd/ReadVariableOp^conv1a/Conv2D/ReadVariableOp^conv1b/BiasAdd/ReadVariableOp^conv1b/Conv2D/ReadVariableOp^conv2a/BiasAdd/ReadVariableOp^conv2a/Conv2D/ReadVariableOp^conv2b/BiasAdd/ReadVariableOp^conv2b/Conv2D/ReadVariableOp^conv2c/BiasAdd/ReadVariableOp^conv2c/Conv2D/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp^dense2/BiasAdd/ReadVariableOp^dense2/MatMul/ReadVariableOp^dense3/BiasAdd/ReadVariableOp^dense3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ À::::::::::::::::2>
conv1a/BiasAdd/ReadVariableOpconv1a/BiasAdd/ReadVariableOp2<
conv1a/Conv2D/ReadVariableOpconv1a/Conv2D/ReadVariableOp2>
conv1b/BiasAdd/ReadVariableOpconv1b/BiasAdd/ReadVariableOp2<
conv1b/Conv2D/ReadVariableOpconv1b/Conv2D/ReadVariableOp2>
conv2a/BiasAdd/ReadVariableOpconv2a/BiasAdd/ReadVariableOp2<
conv2a/Conv2D/ReadVariableOpconv2a/Conv2D/ReadVariableOp2>
conv2b/BiasAdd/ReadVariableOpconv2b/BiasAdd/ReadVariableOp2<
conv2b/Conv2D/ReadVariableOpconv2b/Conv2D/ReadVariableOp2>
conv2c/BiasAdd/ReadVariableOpconv2c/BiasAdd/ReadVariableOp2<
conv2c/Conv2D/ReadVariableOpconv2c/Conv2D/ReadVariableOp2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp2>
dense2/BiasAdd/ReadVariableOpdense2/BiasAdd/ReadVariableOp2<
dense2/MatMul/ReadVariableOpdense2/MatMul/ReadVariableOp2>
dense3/BiasAdd/ReadVariableOpdense3/BiasAdd/ReadVariableOp2<
dense3/MatMul/ReadVariableOpdense3/MatMul/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
 
_user_specified_nameinputs
Ð

Ü
C__inference_conv1a_layer_call_and_return_conditional_losses_1784998

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5}*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5}2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5}2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5}2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿmý::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý
 
_user_specified_nameinputs
Î

Ü
C__inference_conv2b_layer_call_and_return_conditional_losses_1785058

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ï
`
D__inference_process_layer_call_and_return_conditional_losses_1784209

inputs
identity
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    8       2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"       @  2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2þ
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿXÀ*

begin_mask*
end_mask2
strided_slice[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
	truediv/y
truedivRealDivstrided_slice:output:0truediv/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿXÀ2	
truedivS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
add/yk
addAddV2truediv:z:0add/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿXÀ2
addk
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"m   ý   2
resize/size´
resize/ResizeBilinearResizeBilinearadd:z:0resize/size:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý*
half_pixel_centers(2
resize/ResizeBilinearz
RGBToHSVRGBToHSV&resize/ResizeBilinear:resized_images:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý2

RGBToHSV
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceRGBToHSV:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý*
ellipsis_mask2
strided_slice_1u
IdentityIdentitystrided_slice_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ À:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
 
_user_specified_nameinputs
ï
`
D__inference_process_layer_call_and_return_conditional_losses_1784190

inputs
identity
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    8       2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"       @  2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2þ
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿXÀ*

begin_mask*
end_mask2
strided_slice[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
	truediv/y
truedivRealDivstrided_slice:output:0truediv/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿXÀ2	
truedivS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
add/yk
addAddV2truediv:z:0add/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿXÀ2
addk
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"m   ý   2
resize/size´
resize/ResizeBilinearResizeBilinearadd:z:0resize/size:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý*
half_pixel_centers(2
resize/ResizeBilinearz
RGBToHSVRGBToHSV&resize/ResizeBilinear:resized_images:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý2

RGBToHSV
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceRGBToHSV:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý*
ellipsis_mask2
strided_slice_1u
IdentityIdentitystrided_slice_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ À:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
 
_user_specified_nameinputs
¾

Õ
)__inference_model_1_layer_call_fn_1784902

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity¢StatefulPartitionedCall²
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_17845512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*p
_input_shapes_
]:ÿÿÿÿÿÿÿÿÿ À::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
 
_user_specified_nameinputs
ý
}
(__inference_conv1b_layer_call_fn_1785027

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3{*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv1b_layer_call_and_return_conditional_losses_17842602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3{2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ5}::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5}
 
_user_specified_nameinputs
ï
`
D__inference_process_layer_call_and_return_conditional_losses_1784977

inputs
identity
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    8       2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"       @  2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2þ
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿXÀ*

begin_mask*
end_mask2
strided_slice[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
	truediv/y
truedivRealDivstrided_slice:output:0truediv/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿXÀ2	
truedivS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
add/yk
addAddV2truediv:z:0add/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿXÀ2
addk
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"m   ý   2
resize/size´
resize/ResizeBilinearResizeBilinearadd:z:0resize/size:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý*
half_pixel_centers(2
resize/ResizeBilinearz
RGBToHSVRGBToHSV&resize/ResizeBilinear:resized_images:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý2

RGBToHSV
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceRGBToHSV:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý*
ellipsis_mask2
strided_slice_1u
IdentityIdentitystrided_slice_1:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ À:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
 
_user_specified_nameinputs
Î

Ü
C__inference_conv2a_layer_call_and_return_conditional_losses_1785038

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ=::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ=
 
_user_specified_nameinputs
Î

Ü
C__inference_conv1b_layer_call_and_return_conditional_losses_1785018

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3{*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3{2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3{2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ3{2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ5}::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ5}
 
_user_specified_nameinputs
õ*
½
 __inference__traced_save_1785229
file_prefix,
(savev2_conv1a_kernel_read_readvariableop*
&savev2_conv1a_bias_read_readvariableop,
(savev2_conv1b_kernel_read_readvariableop*
&savev2_conv1b_bias_read_readvariableop,
(savev2_conv2a_kernel_read_readvariableop*
&savev2_conv2a_bias_read_readvariableop,
(savev2_conv2b_kernel_read_readvariableop*
&savev2_conv2b_bias_read_readvariableop,
(savev2_conv2c_kernel_read_readvariableop*
&savev2_conv2c_bias_read_readvariableop,
(savev2_dense1_kernel_read_readvariableop*
&savev2_dense1_bias_read_readvariableop,
(savev2_dense2_kernel_read_readvariableop*
&savev2_dense2_bias_read_readvariableop,
(savev2_dense3_kernel_read_readvariableop*
&savev2_dense3_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*£
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesª
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesÚ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv1a_kernel_read_readvariableop&savev2_conv1a_bias_read_readvariableop(savev2_conv1b_kernel_read_readvariableop&savev2_conv1b_bias_read_readvariableop(savev2_conv2a_kernel_read_readvariableop&savev2_conv2a_bias_read_readvariableop(savev2_conv2b_kernel_read_readvariableop&savev2_conv2b_bias_read_readvariableop(savev2_conv2c_kernel_read_readvariableop&savev2_conv2c_bias_read_readvariableop(savev2_dense1_kernel_read_readvariableop&savev2_dense1_bias_read_readvariableop(savev2_dense2_kernel_read_readvariableop&savev2_dense2_bias_read_readvariableop(savev2_dense3_kernel_read_readvariableop&savev2_dense3_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Å
_input_shapes³
°: ::::: : :  : : *:*:
::	:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,	(
&
_output_shapes
: *: 


_output_shapes
:*:&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
»
E
)__inference_process_layer_call_fn_1784987

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_process_layer_call_and_return_conditional_losses_17842092
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿmý2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ À:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ À
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*³
serving_default
E
input_2:
serving_default_input_2:0ÿÿÿÿÿÿÿÿÿ À:
dense30
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¥
æ
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
	optimizer
loss
regularization_losses
trainable_variables
	variables
	keras_api

signatures
__call__
+&call_and_return_all_conditional_losses
_default_save_signature"
_tf_keras_network{"class_name": "Functional", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 160, 320, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Lambda", "config": {"name": "process", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 160, 320, 3]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAgAAAAFAAAAQwAAAHO2AAAAfABqAGQBGQB8AGoAZAIZAAIAfQF9AnQBfAF0\nAhQAgwF9A3QBfAF8AXQDFAAYAIMBfQR0AXwCdAQUAIMBfQV0AXwCfAJ0BRQAGACDAX0GfABkA2QD\nhQJ8A3wEhQJ8BXwGhQJmAxkAfQB0BmoHoAh8AHQGagmhAn0AfABkBHQKdAsYABsAGwB0CxcAfQB0\nBmoMoA18AHQOoQJ9AHQGagygD3wAoQFkBWQBZAaFAmYCGQB9B3wHUwApB/oQSW1hZ2UgcHJvY2Vz\nc2luZ+kBAAAA6QIAAABO6f8AAAAu6QMAAAApENoFc2hhcGXaBXJvdW5k2ghDUk9QX1RPUNoLQ1JP\nUF9CT1RUT03aCUNST1BfTEVGVNoKQ1JPUF9SSUdIVNoCdGbaBmR0eXBlc9oEY2FzdNoHZmxvYXQz\nMtoRVkFMVUVfUkVTQ0FMRV9NQVjaEVZBTFVFX1JFU0NBTEVfTUlO2gVpbWFnZdoGcmVzaXpl2gZS\nRVNJWkXaCnJnYl90b19oc3YpCNoDaW1n2gZoZWlnaHTaBXdpZHRo2gd0b3BfcG9z2gpib3R0b21f\ncG9z2ghsZWZ0X3Bvc9oJcmlnaHRfcG9z2gZzdl9pbWepAHIeAAAA+j9DOi9Vc2Vycy84MDUxL0Rl\nc2t0b3AvU2hpbk5pc2hpbXVyYV9GWVBfYXV0b25vbW91c19jYXIvbW9kZWwucHnaC3Byb2Nlc3Nf\naW1nDAAAAHMWAAAAAAIWAQwBEAEMARACGgEQARQBDgIYAQ==\n", null, null]}, "function_type": "lambda", "module": "model", "output_shape": {"class_name": "__tuple__", "items": [83, 179, 2]}, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "process", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1a", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 83, 179, 2]}, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1a", "inbound_nodes": [[["process", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1b", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 88, 24]}, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1b", "inbound_nodes": [[["conv1a", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "pool1c", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 38, 86, 24]}, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool1c", "inbound_nodes": [[["conv1b", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2a", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 19, 43, 24]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2a", "inbound_nodes": [[["pool1c", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2b", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 20, 32]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2b", "inbound_nodes": [[["conv2a", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2c", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6, 18, 32]}, "dtype": "float32", "filters": 42, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2c", "inbound_nodes": [[["conv2b", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2, 8, 48]}, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["conv2c", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense1", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense2", "inbound_nodes": [[["dense1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense3", "inbound_nodes": [[["dense2", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["dense3", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 160, 320, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 160, 320, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 160, 320, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Lambda", "config": {"name": "process", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 160, 320, 3]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAgAAAAFAAAAQwAAAHO2AAAAfABqAGQBGQB8AGoAZAIZAAIAfQF9AnQBfAF0\nAhQAgwF9A3QBfAF8AXQDFAAYAIMBfQR0AXwCdAQUAIMBfQV0AXwCfAJ0BRQAGACDAX0GfABkA2QD\nhQJ8A3wEhQJ8BXwGhQJmAxkAfQB0BmoHoAh8AHQGagmhAn0AfABkBHQKdAsYABsAGwB0CxcAfQB0\nBmoMoA18AHQOoQJ9AHQGagygD3wAoQFkBWQBZAaFAmYCGQB9B3wHUwApB/oQSW1hZ2UgcHJvY2Vz\nc2luZ+kBAAAA6QIAAABO6f8AAAAu6QMAAAApENoFc2hhcGXaBXJvdW5k2ghDUk9QX1RPUNoLQ1JP\nUF9CT1RUT03aCUNST1BfTEVGVNoKQ1JPUF9SSUdIVNoCdGbaBmR0eXBlc9oEY2FzdNoHZmxvYXQz\nMtoRVkFMVUVfUkVTQ0FMRV9NQVjaEVZBTFVFX1JFU0NBTEVfTUlO2gVpbWFnZdoGcmVzaXpl2gZS\nRVNJWkXaCnJnYl90b19oc3YpCNoDaW1n2gZoZWlnaHTaBXdpZHRo2gd0b3BfcG9z2gpib3R0b21f\ncG9z2ghsZWZ0X3Bvc9oJcmlnaHRfcG9z2gZzdl9pbWepAHIeAAAA+j9DOi9Vc2Vycy84MDUxL0Rl\nc2t0b3AvU2hpbk5pc2hpbXVyYV9GWVBfYXV0b25vbW91c19jYXIvbW9kZWwucHnaC3Byb2Nlc3Nf\naW1nDAAAAHMWAAAAAAIWAQwBEAEMARACGgEQARQBDgIYAQ==\n", null, null]}, "function_type": "lambda", "module": "model", "output_shape": {"class_name": "__tuple__", "items": [83, 179, 2]}, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "process", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1a", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 83, 179, 2]}, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1a", "inbound_nodes": [[["process", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1b", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 88, 24]}, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1b", "inbound_nodes": [[["conv1a", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "pool1c", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 38, 86, 24]}, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "pool1c", "inbound_nodes": [[["conv1b", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2a", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 19, 43, 24]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2a", "inbound_nodes": [[["pool1c", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2b", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 20, 32]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2b", "inbound_nodes": [[["conv2a", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2c", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6, 18, 32]}, "dtype": "float32", "filters": 42, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2c", "inbound_nodes": [[["conv2b", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2, 8, 48]}, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["conv2c", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense1", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense2", "inbound_nodes": [[["dense1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense3", "inbound_nodes": [[["dense2", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["dense3", 0, 0]]}}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.001, "decay": 0.0, "rho": 0.9, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
ý"ú
_tf_keras_input_layerÚ{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 160, 320, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 160, 320, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
õ
	variables
regularization_losses
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"ä
_tf_keras_layerÊ{"class_name": "Lambda", "name": "process", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 160, 320, 3]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "process", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 160, 320, 3]}, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAgAAAAFAAAAQwAAAHO2AAAAfABqAGQBGQB8AGoAZAIZAAIAfQF9AnQBfAF0\nAhQAgwF9A3QBfAF8AXQDFAAYAIMBfQR0AXwCdAQUAIMBfQV0AXwCfAJ0BRQAGACDAX0GfABkA2QD\nhQJ8A3wEhQJ8BXwGhQJmAxkAfQB0BmoHoAh8AHQGagmhAn0AfABkBHQKdAsYABsAGwB0CxcAfQB0\nBmoMoA18AHQOoQJ9AHQGagygD3wAoQFkBWQBZAaFAmYCGQB9B3wHUwApB/oQSW1hZ2UgcHJvY2Vz\nc2luZ+kBAAAA6QIAAABO6f8AAAAu6QMAAAApENoFc2hhcGXaBXJvdW5k2ghDUk9QX1RPUNoLQ1JP\nUF9CT1RUT03aCUNST1BfTEVGVNoKQ1JPUF9SSUdIVNoCdGbaBmR0eXBlc9oEY2FzdNoHZmxvYXQz\nMtoRVkFMVUVfUkVTQ0FMRV9NQVjaEVZBTFVFX1JFU0NBTEVfTUlO2gVpbWFnZdoGcmVzaXpl2gZS\nRVNJWkXaCnJnYl90b19oc3YpCNoDaW1n2gZoZWlnaHTaBXdpZHRo2gd0b3BfcG9z2gpib3R0b21f\ncG9z2ghsZWZ0X3Bvc9oJcmlnaHRfcG9z2gZzdl9pbWepAHIeAAAA+j9DOi9Vc2Vycy84MDUxL0Rl\nc2t0b3AvU2hpbk5pc2hpbXVyYV9GWVBfYXV0b25vbW91c19jYXIvbW9kZWwucHnaC3Byb2Nlc3Nf\naW1nDAAAAHMWAAAAAAIWAQwBEAEMARACGgEQARQBDgIYAQ==\n", null, null]}, "function_type": "lambda", "module": "model", "output_shape": {"class_name": "__tuple__", "items": [83, 179, 2]}, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
ô


kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"Í	
_tf_keras_layer³	{"class_name": "Conv2D", "name": "conv1a", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 83, 179, 2]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1a", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 83, 179, 2]}, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 109, 253, 2]}}
õ


kernel
bias
 	variables
!regularization_losses
"trainable_variables
#	keras_api
__call__
+&call_and_return_all_conditional_losses"Î	
_tf_keras_layer´	{"class_name": "Conv2D", "name": "conv1b", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 88, 24]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1b", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 88, 24]}, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 53, 125, 24]}}
ò
$	variables
%regularization_losses
&trainable_variables
'	keras_api
__call__
+&call_and_return_all_conditional_losses"á
_tf_keras_layerÇ{"class_name": "MaxPooling2D", "name": "pool1c", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 38, 86, 24]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "pool1c", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 38, 86, 24]}, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [3, 3]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ô


(kernel
)bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
__call__
+&call_and_return_all_conditional_losses"Í	
_tf_keras_layer³	{"class_name": "Conv2D", "name": "conv2a", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 19, 43, 24]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2a", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 19, 43, 24]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 25, 61, 24]}}
ò


.kernel
/bias
0	variables
1regularization_losses
2trainable_variables
3	keras_api
__call__
+&call_and_return_all_conditional_losses"Ë	
_tf_keras_layer±	{"class_name": "Conv2D", "name": "conv2b", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 20, 32]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2b", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 8, 20, 32]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 30, 32]}}
ò


4kernel
5bias
6	variables
7regularization_losses
8trainable_variables
9	keras_api
__call__
+&call_and_return_all_conditional_losses"Ë	
_tf_keras_layer±	{"class_name": "Conv2D", "name": "conv2c", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6, 18, 32]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2c", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 6, 18, 32]}, "dtype": "float32", "filters": 42, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 28, 32]}}
ã
:	variables
;regularization_losses
<trainable_variables
=	keras_api
__call__
+&call_and_return_all_conditional_losses"Ò
_tf_keras_layer¸{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2, 8, 48]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2, 8, 48]}, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ê

>kernel
?bias
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
__call__
+ &call_and_return_all_conditional_losses"Ã
_tf_keras_layer©{"class_name": "Dense", "name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 768]}, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2184}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2184]}}
ç

Dkernel
Ebias
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
¡__call__
+¢&call_and_return_all_conditional_losses"À
_tf_keras_layer¦{"class_name": "Dense", "name": "dense2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128]}, "dtype": "float32", "units": 16, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
â

Jkernel
Kbias
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
£__call__
+¤&call_and_return_all_conditional_losses"»
_tf_keras_layer¡{"class_name": "Dense", "name": "dense3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
"
	optimizer
 "
trackable_dict_wrapper
 "
trackable_list_wrapper

0
1
2
3
(4
)5
.6
/7
48
59
>10
?11
D12
E13
J14
K15"
trackable_list_wrapper

0
1
2
3
(4
)5
.6
/7
48
59
>10
?11
D12
E13
J14
K15"
trackable_list_wrapper
Î
Pnon_trainable_variables
regularization_losses

Qlayers
Rlayer_regularization_losses
Smetrics
Tlayer_metrics
trainable_variables
	variables
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
¥serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
Unon_trainable_variables
	variables
regularization_losses

Vlayers
Wmetrics
Xlayer_metrics
trainable_variables
Ylayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
':%2conv1a/kernel
:2conv1a/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
Znon_trainable_variables
	variables
regularization_losses

[layers
\metrics
]layer_metrics
trainable_variables
^layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
':%2conv1b/kernel
:2conv1b/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
_non_trainable_variables
 	variables
!regularization_losses

`layers
ametrics
blayer_metrics
"trainable_variables
clayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
dnon_trainable_variables
$	variables
%regularization_losses

elayers
fmetrics
glayer_metrics
&trainable_variables
hlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
':% 2conv2a/kernel
: 2conv2a/bias
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
°
inon_trainable_variables
*	variables
+regularization_losses

jlayers
kmetrics
llayer_metrics
,trainable_variables
mlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
':%  2conv2b/kernel
: 2conv2b/bias
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
°
nnon_trainable_variables
0	variables
1regularization_losses

olayers
pmetrics
qlayer_metrics
2trainable_variables
rlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
':% *2conv2c/kernel
:*2conv2c/bias
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
°
snon_trainable_variables
6	variables
7regularization_losses

tlayers
umetrics
vlayer_metrics
8trainable_variables
wlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
xnon_trainable_variables
:	variables
;regularization_losses

ylayers
zmetrics
{layer_metrics
<trainable_variables
|layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!:
2dense1/kernel
:2dense1/bias
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
²
}non_trainable_variables
@	variables
Aregularization_losses

~layers
metrics
layer_metrics
Btrainable_variables
 layer_regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
 :	2dense2/kernel
:2dense2/bias
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
µ
non_trainable_variables
F	variables
Gregularization_losses
layers
metrics
layer_metrics
Htrainable_variables
 layer_regularization_losses
¡__call__
+¢&call_and_return_all_conditional_losses
'¢"call_and_return_conditional_losses"
_generic_user_object
:2dense3/kernel
:2dense3/bias
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
µ
non_trainable_variables
L	variables
Mregularization_losses
layers
metrics
layer_metrics
Ntrainable_variables
 layer_regularization_losses
£__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
	8

9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
ò2ï
)__inference_model_1_layer_call_fn_1784939
)__inference_model_1_layer_call_fn_1784902
)__inference_model_1_layer_call_fn_1784586
)__inference_model_1_layer_call_fn_1784670À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
D__inference_model_1_layer_call_and_return_conditional_losses_1784865
D__inference_model_1_layer_call_and_return_conditional_losses_1784787
D__inference_model_1_layer_call_and_return_conditional_losses_1784454
D__inference_model_1_layer_call_and_return_conditional_losses_1784501À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ê2ç
"__inference__wrapped_model_1784155À
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *0¢-
+(
input_2ÿÿÿÿÿÿÿÿÿ À
2
)__inference_process_layer_call_fn_1784987
)__inference_process_layer_call_fn_1784982À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
D__inference_process_layer_call_and_return_conditional_losses_1784977
D__inference_process_layer_call_and_return_conditional_losses_1784958À
·²³
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
(__inference_conv1a_layer_call_fn_1785007¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_conv1a_layer_call_and_return_conditional_losses_1784998¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_conv1b_layer_call_fn_1785027¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_conv1b_layer_call_and_return_conditional_losses_1785018¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
(__inference_pool1c_layer_call_fn_1784167à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
«2¨
C__inference_pool1c_layer_call_and_return_conditional_losses_1784161à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ò2Ï
(__inference_conv2a_layer_call_fn_1785047¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_conv2a_layer_call_and_return_conditional_losses_1785038¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_conv2b_layer_call_fn_1785067¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_conv2b_layer_call_and_return_conditional_losses_1785058¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_conv2c_layer_call_fn_1785087¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_conv2c_layer_call_and_return_conditional_losses_1785078¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_flatten_layer_call_fn_1785098¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_flatten_layer_call_and_return_conditional_losses_1785093¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense1_layer_call_fn_1785118¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense1_layer_call_and_return_conditional_losses_1785109¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense2_layer_call_fn_1785138¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense2_layer_call_and_return_conditional_losses_1785129¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense3_layer_call_fn_1785158¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense3_layer_call_and_return_conditional_losses_1785149¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÌBÉ
%__inference_signature_wrapper_1784709input_2"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ¥
"__inference__wrapped_model_1784155()./45>?DEJK:¢7
0¢-
+(
input_2ÿÿÿÿÿÿÿÿÿ À
ª "/ª,
*
dense3 
dense3ÿÿÿÿÿÿÿÿÿ´
C__inference_conv1a_layer_call_and_return_conditional_losses_1784998m8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿmý
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ5}
 
(__inference_conv1a_layer_call_fn_1785007`8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿmý
ª " ÿÿÿÿÿÿÿÿÿ5}³
C__inference_conv1b_layer_call_and_return_conditional_losses_1785018l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ5}
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ3{
 
(__inference_conv1b_layer_call_fn_1785027_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ5}
ª " ÿÿÿÿÿÿÿÿÿ3{³
C__inference_conv2a_layer_call_and_return_conditional_losses_1785038l()7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ=
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 
(__inference_conv2a_layer_call_fn_1785047_()7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ=
ª " ÿÿÿÿÿÿÿÿÿ ³
C__inference_conv2b_layer_call_and_return_conditional_losses_1785058l./7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
 
(__inference_conv2b_layer_call_fn_1785067_./7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ
 ³
C__inference_conv2c_layer_call_and_return_conditional_losses_1785078l457¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ*
 
(__inference_conv2c_layer_call_fn_1785087_457¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
 
ª " ÿÿÿÿÿÿÿÿÿ*¥
C__inference_dense1_layer_call_and_return_conditional_losses_1785109^>?0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 }
(__inference_dense1_layer_call_fn_1785118Q>?0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
C__inference_dense2_layer_call_and_return_conditional_losses_1785129]DE0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
(__inference_dense2_layer_call_fn_1785138PDE0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ£
C__inference_dense3_layer_call_and_return_conditional_losses_1785149\JK/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_dense3_layer_call_fn_1785158OJK/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
D__inference_flatten_layer_call_and_return_conditional_losses_1785093a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ*
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
)__inference_flatten_layer_call_fn_1785098T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ*
ª "ÿÿÿÿÿÿÿÿÿÅ
D__inference_model_1_layer_call_and_return_conditional_losses_1784454}()./45>?DEJKB¢?
8¢5
+(
input_2ÿÿÿÿÿÿÿÿÿ À
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Å
D__inference_model_1_layer_call_and_return_conditional_losses_1784501}()./45>?DEJKB¢?
8¢5
+(
input_2ÿÿÿÿÿÿÿÿÿ À
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ä
D__inference_model_1_layer_call_and_return_conditional_losses_1784787|()./45>?DEJKA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ À
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ä
D__inference_model_1_layer_call_and_return_conditional_losses_1784865|()./45>?DEJKA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ À
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
)__inference_model_1_layer_call_fn_1784586p()./45>?DEJKB¢?
8¢5
+(
input_2ÿÿÿÿÿÿÿÿÿ À
p

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_model_1_layer_call_fn_1784670p()./45>?DEJKB¢?
8¢5
+(
input_2ÿÿÿÿÿÿÿÿÿ À
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_model_1_layer_call_fn_1784902o()./45>?DEJKA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ À
p

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_model_1_layer_call_fn_1784939o()./45>?DEJKA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ À
p 

 
ª "ÿÿÿÿÿÿÿÿÿæ
C__inference_pool1c_layer_call_and_return_conditional_losses_1784161R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¾
(__inference_pool1c_layer_call_fn_1784167R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ»
D__inference_process_layer_call_and_return_conditional_losses_1784958sA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ À

 
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿmý
 »
D__inference_process_layer_call_and_return_conditional_losses_1784977sA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ À

 
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿmý
 
)__inference_process_layer_call_fn_1784982fA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ À

 
p
ª "!ÿÿÿÿÿÿÿÿÿmý
)__inference_process_layer_call_fn_1784987fA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ À

 
p 
ª "!ÿÿÿÿÿÿÿÿÿmý´
%__inference_signature_wrapper_1784709()./45>?DEJKE¢B
¢ 
;ª8
6
input_2+(
input_2ÿÿÿÿÿÿÿÿÿ À"/ª,
*
dense3 
dense3ÿÿÿÿÿÿÿÿÿ