ив
ьЙ
B
AssignVariableOp
resource
value"dtype"
dtypetypeѕ
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
Џ
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
ѓ
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
delete_old_dirsbool(ѕ
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
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
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
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
Й
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
executor_typestring ѕ
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8ђщ

ё
conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_10/kernel
}
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*&
_output_shapes
: *
dtype0
t
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_10/bias
m
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes
: *
dtype0
ё
conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_11/kernel
}
$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_11/bias
m
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes
:@*
dtype0
Ё
conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ђ*!
shared_nameconv2d_12/kernel
~
$conv2d_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_12/kernel*'
_output_shapes
:@ђ*
dtype0
u
conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_nameconv2d_12/bias
n
"conv2d_12/bias/Read/ReadVariableOpReadVariableOpconv2d_12/bias*
_output_shapes	
:ђ*
dtype0
є
conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*!
shared_nameconv2d_13/kernel

$conv2d_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_13/kernel*(
_output_shapes
:ђђ*
dtype0
u
conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_nameconv2d_13/bias
n
"conv2d_13/bias/Read/ReadVariableOpReadVariableOpconv2d_13/bias*
_output_shapes	
:ђ*
dtype0
{
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ╚ђ*
shared_namedense_6/kernel
t
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*!
_output_shapes
:ђ╚ђ*
dtype0
q
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_namedense_6/bias
j
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes	
:ђ*
dtype0
y
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђd*
shared_namedense_7/kernel
r
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes
:	ђd*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:d*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
њ
Adam/conv2d_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_10/kernel/m
І
+Adam/conv2d_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/kernel/m*&
_output_shapes
: *
dtype0
ѓ
Adam/conv2d_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_10/bias/m
{
)Adam/conv2d_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/bias/m*
_output_shapes
: *
dtype0
њ
Adam/conv2d_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_11/kernel/m
І
+Adam/conv2d_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/kernel/m*&
_output_shapes
: @*
dtype0
ѓ
Adam/conv2d_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_11/bias/m
{
)Adam/conv2d_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/bias/m*
_output_shapes
:@*
dtype0
Њ
Adam/conv2d_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ђ*(
shared_nameAdam/conv2d_12/kernel/m
ї
+Adam/conv2d_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/kernel/m*'
_output_shapes
:@ђ*
dtype0
Ѓ
Adam/conv2d_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_12/bias/m
|
)Adam/conv2d_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/bias/m*
_output_shapes	
:ђ*
dtype0
ћ
Adam/conv2d_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*(
shared_nameAdam/conv2d_13/kernel/m
Ї
+Adam/conv2d_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/kernel/m*(
_output_shapes
:ђђ*
dtype0
Ѓ
Adam/conv2d_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_13/bias/m
|
)Adam/conv2d_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/bias/m*
_output_shapes	
:ђ*
dtype0
Ѕ
Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ╚ђ*&
shared_nameAdam/dense_6/kernel/m
ѓ
)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m*!
_output_shapes
:ђ╚ђ*
dtype0

Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*$
shared_nameAdam/dense_6/bias/m
x
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
_output_shapes	
:ђ*
dtype0
Є
Adam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђd*&
shared_nameAdam/dense_7/kernel/m
ђ
)Adam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/m*
_output_shapes
:	ђd*
dtype0
~
Adam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*$
shared_nameAdam/dense_7/bias/m
w
'Adam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/m*
_output_shapes
:d*
dtype0
њ
Adam/conv2d_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_10/kernel/v
І
+Adam/conv2d_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/kernel/v*&
_output_shapes
: *
dtype0
ѓ
Adam/conv2d_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_10/bias/v
{
)Adam/conv2d_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/bias/v*
_output_shapes
: *
dtype0
њ
Adam/conv2d_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_11/kernel/v
І
+Adam/conv2d_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/kernel/v*&
_output_shapes
: @*
dtype0
ѓ
Adam/conv2d_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_11/bias/v
{
)Adam/conv2d_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/bias/v*
_output_shapes
:@*
dtype0
Њ
Adam/conv2d_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ђ*(
shared_nameAdam/conv2d_12/kernel/v
ї
+Adam/conv2d_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/kernel/v*'
_output_shapes
:@ђ*
dtype0
Ѓ
Adam/conv2d_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_12/bias/v
|
)Adam/conv2d_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/bias/v*
_output_shapes	
:ђ*
dtype0
ћ
Adam/conv2d_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*(
shared_nameAdam/conv2d_13/kernel/v
Ї
+Adam/conv2d_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/kernel/v*(
_output_shapes
:ђђ*
dtype0
Ѓ
Adam/conv2d_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_13/bias/v
|
)Adam/conv2d_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/bias/v*
_output_shapes	
:ђ*
dtype0
Ѕ
Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ╚ђ*&
shared_nameAdam/dense_6/kernel/v
ѓ
)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v*!
_output_shapes
:ђ╚ђ*
dtype0

Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*$
shared_nameAdam/dense_6/bias/v
x
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
_output_shapes	
:ђ*
dtype0
Є
Adam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђd*&
shared_nameAdam/dense_7/kernel/v
ђ
)Adam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/v*
_output_shapes
:	ђd*
dtype0
~
Adam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*$
shared_nameAdam/dense_7/bias/v
w
'Adam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/v*
_output_shapes
:d*
dtype0

NoOpNoOp
й\
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Э[
valueЬ[Bв[ BС[
 
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer-14
layer-15
layer_with_weights-5
layer-16
layer-17
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
 trainable_variables
!	variables
"	keras_api
R
#regularization_losses
$trainable_variables
%	variables
&	keras_api
h

'kernel
(bias
)regularization_losses
*trainable_variables
+	variables
,	keras_api
R
-regularization_losses
.trainable_variables
/	variables
0	keras_api
R
1regularization_losses
2trainable_variables
3	variables
4	keras_api
h

5kernel
6bias
7regularization_losses
8trainable_variables
9	variables
:	keras_api
R
;regularization_losses
<trainable_variables
=	variables
>	keras_api
R
?regularization_losses
@trainable_variables
A	variables
B	keras_api
h

Ckernel
Dbias
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
R
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
R
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
R
Qregularization_losses
Rtrainable_variables
S	variables
T	keras_api
h

Ukernel
Vbias
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
R
[regularization_losses
\trainable_variables
]	variables
^	keras_api
R
_regularization_losses
`trainable_variables
a	variables
b	keras_api
h

ckernel
dbias
eregularization_losses
ftrainable_variables
g	variables
h	keras_api
R
iregularization_losses
jtrainable_variables
k	variables
l	keras_api
░
miter

nbeta_1

obeta_2
	pdecay
qlearning_ratem▄mП'mя(m▀5mЯ6mрCmРDmсUmСVmтcmТdmуvУvж'vЖ(vв5vВ6vьCvЬDv№Uv­VvыcvЫdvз
 
V
0
1
'2
(3
54
65
C6
D7
U8
V9
c10
d11
V
0
1
'2
(3
54
65
C6
D7
U8
V9
c10
d11
Г
rlayer_metrics
snon_trainable_variables
regularization_losses
trainable_variables
	variables
tmetrics

ulayers
vlayer_regularization_losses
 
\Z
VARIABLE_VALUEconv2d_10/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_10/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
Г
wlayer_metrics
xnon_trainable_variables
regularization_losses
trainable_variables
	variables
ymetrics

zlayers
{layer_regularization_losses
 
 
 
«
|layer_metrics
}non_trainable_variables
regularization_losses
 trainable_variables
!	variables
~metrics

layers
 ђlayer_regularization_losses
 
 
 
▓
Ђlayer_metrics
ѓnon_trainable_variables
#regularization_losses
$trainable_variables
%	variables
Ѓmetrics
ёlayers
 Ёlayer_regularization_losses
\Z
VARIABLE_VALUEconv2d_11/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_11/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

'0
(1

'0
(1
▓
єlayer_metrics
Єnon_trainable_variables
)regularization_losses
*trainable_variables
+	variables
ѕmetrics
Ѕlayers
 іlayer_regularization_losses
 
 
 
▓
Іlayer_metrics
їnon_trainable_variables
-regularization_losses
.trainable_variables
/	variables
Їmetrics
јlayers
 Јlayer_regularization_losses
 
 
 
▓
љlayer_metrics
Љnon_trainable_variables
1regularization_losses
2trainable_variables
3	variables
њmetrics
Њlayers
 ћlayer_regularization_losses
\Z
VARIABLE_VALUEconv2d_12/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_12/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

50
61

50
61
▓
Ћlayer_metrics
ќnon_trainable_variables
7regularization_losses
8trainable_variables
9	variables
Ќmetrics
ўlayers
 Ўlayer_regularization_losses
 
 
 
▓
џlayer_metrics
Џnon_trainable_variables
;regularization_losses
<trainable_variables
=	variables
юmetrics
Юlayers
 ъlayer_regularization_losses
 
 
 
▓
Ъlayer_metrics
аnon_trainable_variables
?regularization_losses
@trainable_variables
A	variables
Аmetrics
бlayers
 Бlayer_regularization_losses
\Z
VARIABLE_VALUEconv2d_13/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_13/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

C0
D1

C0
D1
▓
цlayer_metrics
Цnon_trainable_variables
Eregularization_losses
Ftrainable_variables
G	variables
дmetrics
Дlayers
 еlayer_regularization_losses
 
 
 
▓
Еlayer_metrics
фnon_trainable_variables
Iregularization_losses
Jtrainable_variables
K	variables
Фmetrics
гlayers
 Гlayer_regularization_losses
 
 
 
▓
«layer_metrics
»non_trainable_variables
Mregularization_losses
Ntrainable_variables
O	variables
░metrics
▒layers
 ▓layer_regularization_losses
 
 
 
▓
│layer_metrics
┤non_trainable_variables
Qregularization_losses
Rtrainable_variables
S	variables
хmetrics
Хlayers
 иlayer_regularization_losses
ZX
VARIABLE_VALUEdense_6/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_6/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

U0
V1

U0
V1
▓
Иlayer_metrics
╣non_trainable_variables
Wregularization_losses
Xtrainable_variables
Y	variables
║metrics
╗layers
 ╝layer_regularization_losses
 
 
 
▓
йlayer_metrics
Йnon_trainable_variables
[regularization_losses
\trainable_variables
]	variables
┐metrics
└layers
 ┴layer_regularization_losses
 
 
 
▓
┬layer_metrics
├non_trainable_variables
_regularization_losses
`trainable_variables
a	variables
─metrics
┼layers
 кlayer_regularization_losses
ZX
VARIABLE_VALUEdense_7/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_7/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

c0
d1

c0
d1
▓
Кlayer_metrics
╚non_trainable_variables
eregularization_losses
ftrainable_variables
g	variables
╔metrics
╩layers
 ╦layer_regularization_losses
 
 
 
▓
╠layer_metrics
═non_trainable_variables
iregularization_losses
jtrainable_variables
k	variables
╬metrics
¤layers
 лlayer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 

Л0
м1
є
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
12
13
14
15
16
17
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
8

Мtotal

нcount
Н	variables
о	keras_api
I

Оtotal

пcount
┘
_fn_kwargs
┌	variables
█	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

М0
н1

Н	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

О0
п1

┌	variables
}
VARIABLE_VALUEAdam/conv2d_10/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_10/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_11/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_11/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_12/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_12/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_13/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_13/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_6/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_7/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_7/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_10/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_10/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_11/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_11/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_12/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_12/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_13/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_13/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_6/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_7/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_7/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ќ
serving_default_conv2d_10_inputPlaceholder*1
_output_shapes
:         ╚╚*
dtype0*&
shape:         ╚╚
Ћ
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_10_inputconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasconv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *,
f'R%
#__inference_signature_wrapper_93174
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ц
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOp$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp$conv2d_12/kernel/Read/ReadVariableOp"conv2d_12/bias/Read/ReadVariableOp$conv2d_13/kernel/Read/ReadVariableOp"conv2d_13/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_10/kernel/m/Read/ReadVariableOp)Adam/conv2d_10/bias/m/Read/ReadVariableOp+Adam/conv2d_11/kernel/m/Read/ReadVariableOp)Adam/conv2d_11/bias/m/Read/ReadVariableOp+Adam/conv2d_12/kernel/m/Read/ReadVariableOp)Adam/conv2d_12/bias/m/Read/ReadVariableOp+Adam/conv2d_13/kernel/m/Read/ReadVariableOp)Adam/conv2d_13/bias/m/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOp)Adam/dense_7/kernel/m/Read/ReadVariableOp'Adam/dense_7/bias/m/Read/ReadVariableOp+Adam/conv2d_10/kernel/v/Read/ReadVariableOp)Adam/conv2d_10/bias/v/Read/ReadVariableOp+Adam/conv2d_11/kernel/v/Read/ReadVariableOp)Adam/conv2d_11/bias/v/Read/ReadVariableOp+Adam/conv2d_12/kernel/v/Read/ReadVariableOp)Adam/conv2d_12/bias/v/Read/ReadVariableOp+Adam/conv2d_13/kernel/v/Read/ReadVariableOp)Adam/conv2d_13/bias/v/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOp)Adam/dense_7/kernel/v/Read/ReadVariableOp'Adam/dense_7/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *'
f"R 
__inference__traced_save_93715
Џ	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasconv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_10/kernel/mAdam/conv2d_10/bias/mAdam/conv2d_11/kernel/mAdam/conv2d_11/bias/mAdam/conv2d_12/kernel/mAdam/conv2d_12/bias/mAdam/conv2d_13/kernel/mAdam/conv2d_13/bias/mAdam/dense_6/kernel/mAdam/dense_6/bias/mAdam/dense_7/kernel/mAdam/dense_7/bias/mAdam/conv2d_10/kernel/vAdam/conv2d_10/bias/vAdam/conv2d_11/kernel/vAdam/conv2d_11/bias/vAdam/conv2d_12/kernel/vAdam/conv2d_12/bias/vAdam/conv2d_13/kernel/vAdam/conv2d_13/bias/vAdam/dense_6/kernel/vAdam/dense_6/bias/vAdam/dense_7/kernel/vAdam/dense_7/bias/v*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference__traced_restore_93860ЊЦ	
┌
d
H__inference_activation_18_layer_call_and_return_conditional_losses_93427

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:         ..ђ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         ..ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ..ђ:X T
0
_output_shapes
:         ..ђ
 
_user_specified_nameinputs
Г
L
0__inference_max_pooling2d_13_layer_call_fn_92651

inputs
identityВ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_926452
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ф^
и
__inference__traced_save_93715
file_prefix/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableop/
+savev2_conv2d_12_kernel_read_readvariableop-
)savev2_conv2d_12_bias_read_readvariableop/
+savev2_conv2d_13_kernel_read_readvariableop-
)savev2_conv2d_13_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_10_kernel_m_read_readvariableop4
0savev2_adam_conv2d_10_bias_m_read_readvariableop6
2savev2_adam_conv2d_11_kernel_m_read_readvariableop4
0savev2_adam_conv2d_11_bias_m_read_readvariableop6
2savev2_adam_conv2d_12_kernel_m_read_readvariableop4
0savev2_adam_conv2d_12_bias_m_read_readvariableop6
2savev2_adam_conv2d_13_kernel_m_read_readvariableop4
0savev2_adam_conv2d_13_bias_m_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableop4
0savev2_adam_dense_7_kernel_m_read_readvariableop2
.savev2_adam_dense_7_bias_m_read_readvariableop6
2savev2_adam_conv2d_10_kernel_v_read_readvariableop4
0savev2_adam_conv2d_10_bias_v_read_readvariableop6
2savev2_adam_conv2d_11_kernel_v_read_readvariableop4
0savev2_adam_conv2d_11_bias_v_read_readvariableop6
2savev2_adam_conv2d_12_kernel_v_read_readvariableop4
0savev2_adam_conv2d_12_bias_v_read_readvariableop6
2savev2_adam_conv2d_13_kernel_v_read_readvariableop4
0savev2_adam_conv2d_13_bias_v_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableop4
0savev2_adam_dense_7_kernel_v_read_readvariableop2
.savev2_adam_dense_7_bias_v_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
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
Const_1І
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
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename║
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*╠
value┬B┐.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesС
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableop+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop+savev2_conv2d_12_kernel_read_readvariableop)savev2_conv2d_12_bias_read_readvariableop+savev2_conv2d_13_kernel_read_readvariableop)savev2_conv2d_13_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_10_kernel_m_read_readvariableop0savev2_adam_conv2d_10_bias_m_read_readvariableop2savev2_adam_conv2d_11_kernel_m_read_readvariableop0savev2_adam_conv2d_11_bias_m_read_readvariableop2savev2_adam_conv2d_12_kernel_m_read_readvariableop0savev2_adam_conv2d_12_bias_m_read_readvariableop2savev2_adam_conv2d_13_kernel_m_read_readvariableop0savev2_adam_conv2d_13_bias_m_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop0savev2_adam_dense_7_kernel_m_read_readvariableop.savev2_adam_dense_7_bias_m_read_readvariableop2savev2_adam_conv2d_10_kernel_v_read_readvariableop0savev2_adam_conv2d_10_bias_v_read_readvariableop2savev2_adam_conv2d_11_kernel_v_read_readvariableop0savev2_adam_conv2d_11_bias_v_read_readvariableop2savev2_adam_conv2d_12_kernel_v_read_readvariableop0savev2_adam_conv2d_12_bias_v_read_readvariableop2savev2_adam_conv2d_13_kernel_v_read_readvariableop0savev2_adam_conv2d_13_bias_v_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableop0savev2_adam_dense_7_kernel_v_read_readvariableop.savev2_adam_dense_7_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
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

identity_1Identity_1:output:0*╔
_input_shapesи
┤: : : : @:@:@ђ:ђ:ђђ:ђ:ђ╚ђ:ђ:	ђd:d: : : : : : : : : : : : @:@:@ђ:ђ:ђђ:ђ:ђ╚ђ:ђ:	ђd:d: : : @:@:@ђ:ђ:ђђ:ђ:ђ╚ђ:ђ:	ђd:d: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@ђ:!

_output_shapes	
:ђ:.*
(
_output_shapes
:ђђ:!

_output_shapes	
:ђ:'	#
!
_output_shapes
:ђ╚ђ:!


_output_shapes	
:ђ:%!

_output_shapes
:	ђd: 

_output_shapes
:d:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@ђ:!

_output_shapes	
:ђ:.*
(
_output_shapes
:ђђ:!

_output_shapes	
:ђ:'#
!
_output_shapes
:ђ╚ђ:!

_output_shapes	
:ђ:% !

_output_shapes
:	ђd: !

_output_shapes
:d:,"(
&
_output_shapes
: : #

_output_shapes
: :,$(
&
_output_shapes
: @: %

_output_shapes
:@:-&)
'
_output_shapes
:@ђ:!'

_output_shapes	
:ђ:.(*
(
_output_shapes
:ђђ:!)

_output_shapes	
:ђ:'*#
!
_output_shapes
:ђ╚ђ:!+

_output_shapes	
:ђ:%,!

_output_shapes
:	ђd: -

_output_shapes
:d:.

_output_shapes
: 
Њ	
█
B__inference_dense_7_layer_call_and_return_conditional_losses_93538

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2	
BiasAddЋ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Й
I
-__inference_activation_19_layer_call_fn_93461

inputs
identity¤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation_19_layer_call_and_return_conditional_losses_928062
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
б
b
)__inference_dropout_3_layer_call_fn_93523

inputs
identityѕбStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_928802
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ђ
~
)__inference_conv2d_13_layer_call_fn_93451

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_927852
StatefulPartitionedCallЌ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Г
L
0__inference_max_pooling2d_10_layer_call_fn_92615

inputs
identityВ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_926092
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
┌
d
H__inference_activation_19_layer_call_and_return_conditional_losses_93456

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:         ђ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
ъ
I
-__inference_activation_20_layer_call_fn_93501

inputs
identityК
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation_20_layer_call_and_return_conditional_losses_928602
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
┴й
Ъ
!__inference__traced_restore_93860
file_prefix%
!assignvariableop_conv2d_10_kernel%
!assignvariableop_1_conv2d_10_bias'
#assignvariableop_2_conv2d_11_kernel%
!assignvariableop_3_conv2d_11_bias'
#assignvariableop_4_conv2d_12_kernel%
!assignvariableop_5_conv2d_12_bias'
#assignvariableop_6_conv2d_13_kernel%
!assignvariableop_7_conv2d_13_bias%
!assignvariableop_8_dense_6_kernel#
assignvariableop_9_dense_6_bias&
"assignvariableop_10_dense_7_kernel$
 assignvariableop_11_dense_7_bias!
assignvariableop_12_adam_iter#
assignvariableop_13_adam_beta_1#
assignvariableop_14_adam_beta_2"
assignvariableop_15_adam_decay*
&assignvariableop_16_adam_learning_rate
assignvariableop_17_total
assignvariableop_18_count
assignvariableop_19_total_1
assignvariableop_20_count_1/
+assignvariableop_21_adam_conv2d_10_kernel_m-
)assignvariableop_22_adam_conv2d_10_bias_m/
+assignvariableop_23_adam_conv2d_11_kernel_m-
)assignvariableop_24_adam_conv2d_11_bias_m/
+assignvariableop_25_adam_conv2d_12_kernel_m-
)assignvariableop_26_adam_conv2d_12_bias_m/
+assignvariableop_27_adam_conv2d_13_kernel_m-
)assignvariableop_28_adam_conv2d_13_bias_m-
)assignvariableop_29_adam_dense_6_kernel_m+
'assignvariableop_30_adam_dense_6_bias_m-
)assignvariableop_31_adam_dense_7_kernel_m+
'assignvariableop_32_adam_dense_7_bias_m/
+assignvariableop_33_adam_conv2d_10_kernel_v-
)assignvariableop_34_adam_conv2d_10_bias_v/
+assignvariableop_35_adam_conv2d_11_kernel_v-
)assignvariableop_36_adam_conv2d_11_bias_v/
+assignvariableop_37_adam_conv2d_12_kernel_v-
)assignvariableop_38_adam_conv2d_12_bias_v/
+assignvariableop_39_adam_conv2d_13_kernel_v-
)assignvariableop_40_adam_conv2d_13_bias_v-
)assignvariableop_41_adam_dense_6_kernel_v+
'assignvariableop_42_adam_dense_6_bias_v-
)assignvariableop_43_adam_dense_7_kernel_v+
'assignvariableop_44_adam_dense_7_bias_v
identity_46ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_42бAssignVariableOp_43бAssignVariableOp_44бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9└
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*╠
value┬B┐.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЖ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesћ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*╬
_output_shapes╗
И::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityа
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_10_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1д
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_10_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2е
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_11_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3д
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_11_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4е
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_12_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5д
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_12_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6е
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_13_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7д
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_13_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8д
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_6_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9ц
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_6_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10ф
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_7_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11е
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_7_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12Ц
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Д
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Д
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15д
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16«
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17А
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18А
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Б
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Б
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21│
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_conv2d_10_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22▒
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_conv2d_10_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23│
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_conv2d_11_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24▒
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_conv2d_11_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25│
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_conv2d_12_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26▒
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_conv2d_12_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27│
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv2d_13_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28▒
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv2d_13_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29▒
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_6_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30»
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_6_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31▒
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_7_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32»
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_7_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33│
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_10_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34▒
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_10_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35│
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_11_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36▒
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_11_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37│
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv2d_12_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38▒
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv2d_12_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39│
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv2d_13_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40▒
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv2d_13_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41▒
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_6_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42»
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_6_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43▒
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_7_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44»
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_7_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_449
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp╝
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_45»
Identity_46IdentityIdentity_45:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_46"#
identity_46Identity_46:output:0*╦
_input_shapes╣
Х: :::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
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
њF
и
G__inference_sequential_3_layer_call_and_return_conditional_losses_92984
conv2d_10_input
conv2d_10_92941
conv2d_10_92943
conv2d_11_92948
conv2d_11_92950
conv2d_12_92955
conv2d_12_92957
conv2d_13_92962
conv2d_13_92964
dense_6_92970
dense_6_92972
dense_7_92977
dense_7_92979
identityѕб!conv2d_10/StatefulPartitionedCallб!conv2d_11/StatefulPartitionedCallб!conv2d_12/StatefulPartitionedCallб!conv2d_13/StatefulPartitionedCallбdense_6/StatefulPartitionedCallбdense_7/StatefulPartitionedCallЕ
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallconv2d_10_inputconv2d_10_92941conv2d_10_92943*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         кк *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_926652#
!conv2d_10/StatefulPartitionedCallљ
activation_16/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         кк * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation_16_layer_call_and_return_conditional_losses_926862
activation_16/PartitionedCallЊ
 max_pooling2d_10/PartitionedCallPartitionedCall&activation_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         cc * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_926092"
 max_pooling2d_10/PartitionedCall┴
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0conv2d_11_92948conv2d_11_92950*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         aa@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_927052#
!conv2d_11/StatefulPartitionedCallј
activation_17/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         aa@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation_17_layer_call_and_return_conditional_losses_927262
activation_17/PartitionedCallЊ
 max_pooling2d_11/PartitionedCallPartitionedCall&activation_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         00@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_926212"
 max_pooling2d_11/PartitionedCall┬
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_11/PartitionedCall:output:0conv2d_12_92955conv2d_12_92957*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ..ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_927452#
!conv2d_12/StatefulPartitionedCallЈ
activation_18/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ..ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation_18_layer_call_and_return_conditional_losses_927662
activation_18/PartitionedCallћ
 max_pooling2d_12/PartitionedCallPartitionedCall&activation_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_926332"
 max_pooling2d_12/PartitionedCall┬
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_12/PartitionedCall:output:0conv2d_13_92962conv2d_13_92964*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_927852#
!conv2d_13/StatefulPartitionedCallЈ
activation_19/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation_19_layer_call_and_return_conditional_losses_928062
activation_19/PartitionedCallћ
 max_pooling2d_13/PartitionedCallPartitionedCall&activation_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         

ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_926452"
 max_pooling2d_13/PartitionedCallч
flatten_3/PartitionedCallPartitionedCall)max_pooling2d_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ђ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_928212
flatten_3/PartitionedCallЕ
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_92970dense_6_92972*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_928392!
dense_6/StatefulPartitionedCallЁ
activation_20/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation_20_layer_call_and_return_conditional_losses_928602
activation_20/PartitionedCallэ
dropout_3/PartitionedCallPartitionedCall&activation_20/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_928852
dropout_3/PartitionedCallе
dense_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_7_92977dense_7_92979*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_929082!
dense_7/StatefulPartitionedCallё
activation_21/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation_21_layer_call_and_return_conditional_losses_929292
activation_21/PartitionedCall╬
IdentityIdentity&activation_21/PartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         ╚╚::::::::::::2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:b ^
1
_output_shapes
:         ╚╚
)
_user_specified_nameconv2d_10_input
н	
А
,__inference_sequential_3_layer_call_fn_93135
conv2d_10_input
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

unknown_10
identityѕбStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallconv2d_10_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_931082
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         ╚╚::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:         ╚╚
)
_user_specified_nameconv2d_10_input
з	
П
D__inference_conv2d_13_layer_call_and_return_conditional_losses_92785

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02
Conv2D/ReadVariableOpЦ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingVALID*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЅ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2	
BiasAddъ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
е
E
)__inference_flatten_3_layer_call_fn_93472

inputs
identity─
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ђ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_928212
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:         ђ╚2

Identity"
identityIdentity:output:0*/
_input_shapes
:         

ђ:X T
0
_output_shapes
:         

ђ
 
_user_specified_nameinputs
Й
I
-__inference_activation_18_layer_call_fn_93432

inputs
identity¤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ..ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation_18_layer_call_and_return_conditional_losses_927662
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ..ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ..ђ:X T
0
_output_shapes
:         ..ђ
 
_user_specified_nameinputs
╣	
ў
,__inference_sequential_3_layer_call_fn_93316

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

unknown_10
identityѕбStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_930332
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         ╚╚::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ╚╚
 
_user_specified_nameinputs
­	
П
D__inference_conv2d_12_layer_call_and_return_conditional_losses_92745

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpќ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02
Conv2D/ReadVariableOpЦ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ..ђ*
paddingVALID*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЅ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ..ђ2	
BiasAddъ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         ..ђ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         00@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         00@
 
_user_specified_nameinputs
я
d
H__inference_activation_16_layer_call_and_return_conditional_losses_92686

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:         кк 2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:         кк 2

Identity"
identityIdentity:output:0*0
_input_shapes
:         кк :Y U
1
_output_shapes
:         кк 
 
_user_specified_nameinputs
┌
d
H__inference_activation_18_layer_call_and_return_conditional_losses_92766

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:         ..ђ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         ..ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ..ђ:X T
0
_output_shapes
:         ..ђ
 
_user_specified_nameinputs
я
|
'__inference_dense_6_layer_call_fn_93491

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_928392
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ђ╚::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:         ђ╚
 
_user_specified_nameinputs
┌
d
H__inference_activation_19_layer_call_and_return_conditional_losses_92806

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:         ђ2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Њ	
█
B__inference_dense_7_layer_call_and_return_conditional_losses_92908

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђd*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2	
BiasAddЋ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
ДG
м
G__inference_sequential_3_layer_call_and_return_conditional_losses_93033

inputs
conv2d_10_92990
conv2d_10_92992
conv2d_11_92997
conv2d_11_92999
conv2d_12_93004
conv2d_12_93006
conv2d_13_93011
conv2d_13_93013
dense_6_93019
dense_6_93021
dense_7_93026
dense_7_93028
identityѕб!conv2d_10/StatefulPartitionedCallб!conv2d_11/StatefulPartitionedCallб!conv2d_12/StatefulPartitionedCallб!conv2d_13/StatefulPartitionedCallбdense_6/StatefulPartitionedCallбdense_7/StatefulPartitionedCallб!dropout_3/StatefulPartitionedCallа
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_92990conv2d_10_92992*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         кк *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_926652#
!conv2d_10/StatefulPartitionedCallљ
activation_16/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         кк * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation_16_layer_call_and_return_conditional_losses_926862
activation_16/PartitionedCallЊ
 max_pooling2d_10/PartitionedCallPartitionedCall&activation_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         cc * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_926092"
 max_pooling2d_10/PartitionedCall┴
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0conv2d_11_92997conv2d_11_92999*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         aa@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_927052#
!conv2d_11/StatefulPartitionedCallј
activation_17/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         aa@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation_17_layer_call_and_return_conditional_losses_927262
activation_17/PartitionedCallЊ
 max_pooling2d_11/PartitionedCallPartitionedCall&activation_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         00@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_926212"
 max_pooling2d_11/PartitionedCall┬
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_11/PartitionedCall:output:0conv2d_12_93004conv2d_12_93006*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ..ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_927452#
!conv2d_12/StatefulPartitionedCallЈ
activation_18/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ..ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation_18_layer_call_and_return_conditional_losses_927662
activation_18/PartitionedCallћ
 max_pooling2d_12/PartitionedCallPartitionedCall&activation_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_926332"
 max_pooling2d_12/PartitionedCall┬
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_12/PartitionedCall:output:0conv2d_13_93011conv2d_13_93013*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_927852#
!conv2d_13/StatefulPartitionedCallЈ
activation_19/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation_19_layer_call_and_return_conditional_losses_928062
activation_19/PartitionedCallћ
 max_pooling2d_13/PartitionedCallPartitionedCall&activation_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         

ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_926452"
 max_pooling2d_13/PartitionedCallч
flatten_3/PartitionedCallPartitionedCall)max_pooling2d_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ђ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_928212
flatten_3/PartitionedCallЕ
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_93019dense_6_93021*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_928392!
dense_6/StatefulPartitionedCallЁ
activation_20/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation_20_layer_call_and_return_conditional_losses_928602
activation_20/PartitionedCallЈ
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall&activation_20/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_928802#
!dropout_3/StatefulPartitionedCall░
dense_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_7_93026dense_7_93028*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_929082!
dense_7/StatefulPartitionedCallё
activation_21/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation_21_layer_call_and_return_conditional_losses_929292
activation_21/PartitionedCallЫ
IdentityIdentity&activation_21/PartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         ╚╚::::::::::::2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:Y U
1
_output_shapes
:         ╚╚
 
_user_specified_nameinputs
Ѕ
c
D__inference_dropout_3_layer_call_and_return_conditional_losses_92880

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeх
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
­	
П
D__inference_conv2d_12_layer_call_and_return_conditional_losses_93413

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpќ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02
Conv2D/ReadVariableOpЦ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ..ђ*
paddingVALID*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЅ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ..ђ2	
BiasAddъ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         ..ђ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         00@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         00@
 
_user_specified_nameinputs
Ђ
g
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_92621

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ш	
П
D__inference_conv2d_10_layer_call_and_return_conditional_losses_93355

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         кк *
paddingVALID*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpі
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         кк 2	
BiasAddЪ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:         кк 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:         ╚╚::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ╚╚
 
_user_specified_nameinputs
ц	
ў
#__inference_signature_wrapper_93174
conv2d_10_input
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

unknown_10
identityѕбStatefulPartitionedCall▄
StatefulPartitionedCallStatefulPartitionedCallconv2d_10_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *)
f$R"
 __inference__wrapped_model_926032
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         ╚╚::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:         ╚╚
)
_user_specified_nameconv2d_10_input
о
d
H__inference_activation_17_layer_call_and_return_conditional_losses_92726

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:         aa@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         aa@2

Identity"
identityIdentity:output:0*.
_input_shapes
:         aa@:W S
/
_output_shapes
:         aa@
 
_user_specified_nameinputs
║
d
H__inference_activation_20_layer_call_and_return_conditional_losses_93496

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:         ђ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
в	
П
D__inference_conv2d_11_layer_call_and_return_conditional_losses_92705

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         aa@*
paddingVALID*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         aa@2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         aa@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         cc ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         cc 
 
_user_specified_nameinputs
║
d
H__inference_activation_20_layer_call_and_return_conditional_losses_92860

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:         ђ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
┬G
█
G__inference_sequential_3_layer_call_and_return_conditional_losses_92938
conv2d_10_input
conv2d_10_92676
conv2d_10_92678
conv2d_11_92716
conv2d_11_92718
conv2d_12_92756
conv2d_12_92758
conv2d_13_92796
conv2d_13_92798
dense_6_92850
dense_6_92852
dense_7_92919
dense_7_92921
identityѕб!conv2d_10/StatefulPartitionedCallб!conv2d_11/StatefulPartitionedCallб!conv2d_12/StatefulPartitionedCallб!conv2d_13/StatefulPartitionedCallбdense_6/StatefulPartitionedCallбdense_7/StatefulPartitionedCallб!dropout_3/StatefulPartitionedCallЕ
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallconv2d_10_inputconv2d_10_92676conv2d_10_92678*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         кк *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_926652#
!conv2d_10/StatefulPartitionedCallљ
activation_16/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         кк * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation_16_layer_call_and_return_conditional_losses_926862
activation_16/PartitionedCallЊ
 max_pooling2d_10/PartitionedCallPartitionedCall&activation_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         cc * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_926092"
 max_pooling2d_10/PartitionedCall┴
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0conv2d_11_92716conv2d_11_92718*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         aa@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_927052#
!conv2d_11/StatefulPartitionedCallј
activation_17/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         aa@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation_17_layer_call_and_return_conditional_losses_927262
activation_17/PartitionedCallЊ
 max_pooling2d_11/PartitionedCallPartitionedCall&activation_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         00@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_926212"
 max_pooling2d_11/PartitionedCall┬
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_11/PartitionedCall:output:0conv2d_12_92756conv2d_12_92758*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ..ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_927452#
!conv2d_12/StatefulPartitionedCallЈ
activation_18/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ..ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation_18_layer_call_and_return_conditional_losses_927662
activation_18/PartitionedCallћ
 max_pooling2d_12/PartitionedCallPartitionedCall&activation_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_926332"
 max_pooling2d_12/PartitionedCall┬
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_12/PartitionedCall:output:0conv2d_13_92796conv2d_13_92798*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_927852#
!conv2d_13/StatefulPartitionedCallЈ
activation_19/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation_19_layer_call_and_return_conditional_losses_928062
activation_19/PartitionedCallћ
 max_pooling2d_13/PartitionedCallPartitionedCall&activation_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         

ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_926452"
 max_pooling2d_13/PartitionedCallч
flatten_3/PartitionedCallPartitionedCall)max_pooling2d_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ђ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_928212
flatten_3/PartitionedCallЕ
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_92850dense_6_92852*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_928392!
dense_6/StatefulPartitionedCallЁ
activation_20/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation_20_layer_call_and_return_conditional_losses_928602
activation_20/PartitionedCallЈ
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall&activation_20/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_928802#
!dropout_3/StatefulPartitionedCall░
dense_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_7_92919dense_7_92921*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_929082!
dense_7/StatefulPartitionedCallё
activation_21/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation_21_layer_call_and_return_conditional_losses_929292
activation_21/PartitionedCallЫ
IdentityIdentity&activation_21/PartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         ╚╚::::::::::::2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:b ^
1
_output_shapes
:         ╚╚
)
_user_specified_nameconv2d_10_input
Ђ
g
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_92633

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╦
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_93518

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
з	
П
D__inference_conv2d_13_layer_call_and_return_conditional_losses_93442

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02
Conv2D/ReadVariableOpЦ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingVALID*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЅ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2	
BiasAddъ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
┌
|
'__inference_dense_7_layer_call_fn_93547

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_929082
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ђ
g
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_92645

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Џ	
█
B__inference_dense_6_layer_call_and_return_conditional_losses_93482

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpљ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:ђ╚ђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddќ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ђ╚::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:         ђ╚
 
_user_specified_nameinputs
Ч
~
)__inference_conv2d_11_layer_call_fn_93393

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         aa@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_927052
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         aa@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         cc ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         cc 
 
_user_specified_nameinputs
ёP
б
G__inference_sequential_3_layer_call_and_return_conditional_losses_93234

inputs,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource,
(conv2d_11_conv2d_readvariableop_resource-
)conv2d_11_biasadd_readvariableop_resource,
(conv2d_12_conv2d_readvariableop_resource-
)conv2d_12_biasadd_readvariableop_resource,
(conv2d_13_conv2d_readvariableop_resource-
)conv2d_13_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource
identityѕб conv2d_10/BiasAdd/ReadVariableOpбconv2d_10/Conv2D/ReadVariableOpб conv2d_11/BiasAdd/ReadVariableOpбconv2d_11/Conv2D/ReadVariableOpб conv2d_12/BiasAdd/ReadVariableOpбconv2d_12/Conv2D/ReadVariableOpб conv2d_13/BiasAdd/ReadVariableOpбconv2d_13/Conv2D/ReadVariableOpбdense_6/BiasAdd/ReadVariableOpбdense_6/MatMul/ReadVariableOpбdense_7/BiasAdd/ReadVariableOpбdense_7/MatMul/ReadVariableOp│
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_10/Conv2D/ReadVariableOp─
conv2d_10/Conv2DConv2Dinputs'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         кк *
paddingVALID*
strides
2
conv2d_10/Conv2Dф
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp▓
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         кк 2
conv2d_10/BiasAddѕ
activation_16/ReluReluconv2d_10/BiasAdd:output:0*
T0*1
_output_shapes
:         кк 2
activation_16/Relu╬
max_pooling2d_10/MaxPoolMaxPool activation_16/Relu:activations:0*/
_output_shapes
:         cc *
ksize
*
paddingVALID*
strides
2
max_pooling2d_10/MaxPool│
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_11/Conv2D/ReadVariableOpП
conv2d_11/Conv2DConv2D!max_pooling2d_10/MaxPool:output:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         aa@*
paddingVALID*
strides
2
conv2d_11/Conv2Dф
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp░
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         aa@2
conv2d_11/BiasAddє
activation_17/ReluReluconv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:         aa@2
activation_17/Relu╬
max_pooling2d_11/MaxPoolMaxPool activation_17/Relu:activations:0*/
_output_shapes
:         00@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_11/MaxPool┤
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02!
conv2d_12/Conv2D/ReadVariableOpя
conv2d_12/Conv2DConv2D!max_pooling2d_11/MaxPool:output:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ..ђ*
paddingVALID*
strides
2
conv2d_12/Conv2DФ
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp▒
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ..ђ2
conv2d_12/BiasAddЄ
activation_18/ReluReluconv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:         ..ђ2
activation_18/Relu¤
max_pooling2d_12/MaxPoolMaxPool activation_18/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_12/MaxPoolх
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_13/Conv2D/ReadVariableOpя
conv2d_13/Conv2DConv2D!max_pooling2d_12/MaxPool:output:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingVALID*
strides
2
conv2d_13/Conv2DФ
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp▒
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_13/BiasAddЄ
activation_19/ReluReluconv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
activation_19/Relu¤
max_pooling2d_13/MaxPoolMaxPool activation_19/Relu:activations:0*0
_output_shapes
:         

ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_13/MaxPools
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"     d  2
flatten_3/Constб
flatten_3/ReshapeReshape!max_pooling2d_13/MaxPool:output:0flatten_3/Const:output:0*
T0*)
_output_shapes
:         ђ╚2
flatten_3/Reshapeе
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*!
_output_shapes
:ђ╚ђ*
dtype02
dense_6/MatMul/ReadVariableOpа
dense_6/MatMulMatMulflatten_3/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_6/MatMulЦ
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02 
dense_6/BiasAdd/ReadVariableOpб
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_6/BiasAdd}
activation_20/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
activation_20/Reluw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_3/dropout/Constг
dropout_3/dropout/MulMul activation_20/Relu:activations:0 dropout_3/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout_3/dropout/Mulѓ
dropout_3/dropout/ShapeShape activation_20/Relu:activations:0*
T0*
_output_shapes
:2
dropout_3/dropout/ShapeМ
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype020
.dropout_3/dropout/random_uniform/RandomUniformЅ
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_3/dropout/GreaterEqual/yу
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2 
dropout_3/dropout/GreaterEqualъ
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout_3/dropout/CastБ
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout_3/dropout/Mul_1д
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	ђd*
dtype02
dense_7/MatMul/ReadVariableOpа
dense_7/MatMulMatMuldropout_3/dropout/Mul_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
dense_7/MatMulц
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02 
dense_7/BiasAdd/ReadVariableOpА
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
dense_7/BiasAddЁ
activation_21/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:         d2
activation_21/SoftmaxЅ
IdentityIdentityactivation_21/Softmax:softmax:0!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         ╚╚::::::::::::2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:Y U
1
_output_shapes
:         ╚╚
 
_user_specified_nameinputs
эE
«
G__inference_sequential_3_layer_call_and_return_conditional_losses_93108

inputs
conv2d_10_93065
conv2d_10_93067
conv2d_11_93072
conv2d_11_93074
conv2d_12_93079
conv2d_12_93081
conv2d_13_93086
conv2d_13_93088
dense_6_93094
dense_6_93096
dense_7_93101
dense_7_93103
identityѕб!conv2d_10/StatefulPartitionedCallб!conv2d_11/StatefulPartitionedCallб!conv2d_12/StatefulPartitionedCallб!conv2d_13/StatefulPartitionedCallбdense_6/StatefulPartitionedCallбdense_7/StatefulPartitionedCallа
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_10_93065conv2d_10_93067*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         кк *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_926652#
!conv2d_10/StatefulPartitionedCallљ
activation_16/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         кк * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation_16_layer_call_and_return_conditional_losses_926862
activation_16/PartitionedCallЊ
 max_pooling2d_10/PartitionedCallPartitionedCall&activation_16/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         cc * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_926092"
 max_pooling2d_10/PartitionedCall┴
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0conv2d_11_93072conv2d_11_93074*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         aa@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_927052#
!conv2d_11/StatefulPartitionedCallј
activation_17/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         aa@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation_17_layer_call_and_return_conditional_losses_927262
activation_17/PartitionedCallЊ
 max_pooling2d_11/PartitionedCallPartitionedCall&activation_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         00@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_926212"
 max_pooling2d_11/PartitionedCall┬
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_11/PartitionedCall:output:0conv2d_12_93079conv2d_12_93081*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ..ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_927452#
!conv2d_12/StatefulPartitionedCallЈ
activation_18/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ..ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation_18_layer_call_and_return_conditional_losses_927662
activation_18/PartitionedCallћ
 max_pooling2d_12/PartitionedCallPartitionedCall&activation_18/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_926332"
 max_pooling2d_12/PartitionedCall┬
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_12/PartitionedCall:output:0conv2d_13_93086conv2d_13_93088*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_13_layer_call_and_return_conditional_losses_927852#
!conv2d_13/StatefulPartitionedCallЈ
activation_19/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation_19_layer_call_and_return_conditional_losses_928062
activation_19/PartitionedCallћ
 max_pooling2d_13/PartitionedCallPartitionedCall&activation_19/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         

ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_926452"
 max_pooling2d_13/PartitionedCallч
flatten_3/PartitionedCallPartitionedCall)max_pooling2d_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ђ╚* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_928212
flatten_3/PartitionedCallЕ
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_93094dense_6_93096*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_928392!
dense_6/StatefulPartitionedCallЁ
activation_20/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation_20_layer_call_and_return_conditional_losses_928602
activation_20/PartitionedCallэ
dropout_3/PartitionedCallPartitionedCall&activation_20/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_928852
dropout_3/PartitionedCallе
dense_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_7_93101dense_7_93103*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_929082!
dense_7/StatefulPartitionedCallё
activation_21/PartitionedCallPartitionedCall(dense_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation_21_layer_call_and_return_conditional_losses_929292
activation_21/PartitionedCall╬
IdentityIdentity&activation_21/PartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         ╚╚::::::::::::2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:Y U
1
_output_shapes
:         ╚╚
 
_user_specified_nameinputs
ш	
П
D__inference_conv2d_10_layer_call_and_return_conditional_losses_92665

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         кк *
paddingVALID*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpі
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         кк 2	
BiasAddЪ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:         кк 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:         ╚╚::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ╚╚
 
_user_specified_nameinputs
ЉY
╝

 __inference__wrapped_model_92603
conv2d_10_input9
5sequential_3_conv2d_10_conv2d_readvariableop_resource:
6sequential_3_conv2d_10_biasadd_readvariableop_resource9
5sequential_3_conv2d_11_conv2d_readvariableop_resource:
6sequential_3_conv2d_11_biasadd_readvariableop_resource9
5sequential_3_conv2d_12_conv2d_readvariableop_resource:
6sequential_3_conv2d_12_biasadd_readvariableop_resource9
5sequential_3_conv2d_13_conv2d_readvariableop_resource:
6sequential_3_conv2d_13_biasadd_readvariableop_resource7
3sequential_3_dense_6_matmul_readvariableop_resource8
4sequential_3_dense_6_biasadd_readvariableop_resource7
3sequential_3_dense_7_matmul_readvariableop_resource8
4sequential_3_dense_7_biasadd_readvariableop_resource
identityѕб-sequential_3/conv2d_10/BiasAdd/ReadVariableOpб,sequential_3/conv2d_10/Conv2D/ReadVariableOpб-sequential_3/conv2d_11/BiasAdd/ReadVariableOpб,sequential_3/conv2d_11/Conv2D/ReadVariableOpб-sequential_3/conv2d_12/BiasAdd/ReadVariableOpб,sequential_3/conv2d_12/Conv2D/ReadVariableOpб-sequential_3/conv2d_13/BiasAdd/ReadVariableOpб,sequential_3/conv2d_13/Conv2D/ReadVariableOpб+sequential_3/dense_6/BiasAdd/ReadVariableOpб*sequential_3/dense_6/MatMul/ReadVariableOpб+sequential_3/dense_7/BiasAdd/ReadVariableOpб*sequential_3/dense_7/MatMul/ReadVariableOp┌
,sequential_3/conv2d_10/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_3/conv2d_10/Conv2D/ReadVariableOpЗ
sequential_3/conv2d_10/Conv2DConv2Dconv2d_10_input4sequential_3/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         кк *
paddingVALID*
strides
2
sequential_3/conv2d_10/Conv2DЛ
-sequential_3/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_3/conv2d_10/BiasAdd/ReadVariableOpТ
sequential_3/conv2d_10/BiasAddBiasAdd&sequential_3/conv2d_10/Conv2D:output:05sequential_3/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         кк 2 
sequential_3/conv2d_10/BiasAdd»
sequential_3/activation_16/ReluRelu'sequential_3/conv2d_10/BiasAdd:output:0*
T0*1
_output_shapes
:         кк 2!
sequential_3/activation_16/Reluш
%sequential_3/max_pooling2d_10/MaxPoolMaxPool-sequential_3/activation_16/Relu:activations:0*/
_output_shapes
:         cc *
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_10/MaxPool┌
,sequential_3/conv2d_11/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02.
,sequential_3/conv2d_11/Conv2D/ReadVariableOpЉ
sequential_3/conv2d_11/Conv2DConv2D.sequential_3/max_pooling2d_10/MaxPool:output:04sequential_3/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         aa@*
paddingVALID*
strides
2
sequential_3/conv2d_11/Conv2DЛ
-sequential_3/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_3/conv2d_11/BiasAdd/ReadVariableOpС
sequential_3/conv2d_11/BiasAddBiasAdd&sequential_3/conv2d_11/Conv2D:output:05sequential_3/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         aa@2 
sequential_3/conv2d_11/BiasAddГ
sequential_3/activation_17/ReluRelu'sequential_3/conv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:         aa@2!
sequential_3/activation_17/Reluш
%sequential_3/max_pooling2d_11/MaxPoolMaxPool-sequential_3/activation_17/Relu:activations:0*/
_output_shapes
:         00@*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_11/MaxPool█
,sequential_3/conv2d_12/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_12_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02.
,sequential_3/conv2d_12/Conv2D/ReadVariableOpњ
sequential_3/conv2d_12/Conv2DConv2D.sequential_3/max_pooling2d_11/MaxPool:output:04sequential_3/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ..ђ*
paddingVALID*
strides
2
sequential_3/conv2d_12/Conv2Dм
-sequential_3/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_3/conv2d_12/BiasAdd/ReadVariableOpт
sequential_3/conv2d_12/BiasAddBiasAdd&sequential_3/conv2d_12/Conv2D:output:05sequential_3/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ..ђ2 
sequential_3/conv2d_12/BiasAdd«
sequential_3/activation_18/ReluRelu'sequential_3/conv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:         ..ђ2!
sequential_3/activation_18/ReluШ
%sequential_3/max_pooling2d_12/MaxPoolMaxPool-sequential_3/activation_18/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_12/MaxPool▄
,sequential_3/conv2d_13/Conv2D/ReadVariableOpReadVariableOp5sequential_3_conv2d_13_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02.
,sequential_3/conv2d_13/Conv2D/ReadVariableOpњ
sequential_3/conv2d_13/Conv2DConv2D.sequential_3/max_pooling2d_12/MaxPool:output:04sequential_3/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingVALID*
strides
2
sequential_3/conv2d_13/Conv2Dм
-sequential_3/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp6sequential_3_conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_3/conv2d_13/BiasAdd/ReadVariableOpт
sequential_3/conv2d_13/BiasAddBiasAdd&sequential_3/conv2d_13/Conv2D:output:05sequential_3/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2 
sequential_3/conv2d_13/BiasAdd«
sequential_3/activation_19/ReluRelu'sequential_3/conv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2!
sequential_3/activation_19/ReluШ
%sequential_3/max_pooling2d_13/MaxPoolMaxPool-sequential_3/activation_19/Relu:activations:0*0
_output_shapes
:         

ђ*
ksize
*
paddingVALID*
strides
2'
%sequential_3/max_pooling2d_13/MaxPoolЇ
sequential_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"     d  2
sequential_3/flatten_3/Constо
sequential_3/flatten_3/ReshapeReshape.sequential_3/max_pooling2d_13/MaxPool:output:0%sequential_3/flatten_3/Const:output:0*
T0*)
_output_shapes
:         ђ╚2 
sequential_3/flatten_3/Reshape¤
*sequential_3/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_6_matmul_readvariableop_resource*!
_output_shapes
:ђ╚ђ*
dtype02,
*sequential_3/dense_6/MatMul/ReadVariableOpн
sequential_3/dense_6/MatMulMatMul'sequential_3/flatten_3/Reshape:output:02sequential_3/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_3/dense_6/MatMul╠
+sequential_3/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02-
+sequential_3/dense_6/BiasAdd/ReadVariableOpо
sequential_3/dense_6/BiasAddBiasAdd%sequential_3/dense_6/MatMul:product:03sequential_3/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_3/dense_6/BiasAddц
sequential_3/activation_20/ReluRelu%sequential_3/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2!
sequential_3/activation_20/Relu░
sequential_3/dropout_3/IdentityIdentity-sequential_3/activation_20/Relu:activations:0*
T0*(
_output_shapes
:         ђ2!
sequential_3/dropout_3/Identity═
*sequential_3/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_3_dense_7_matmul_readvariableop_resource*
_output_shapes
:	ђd*
dtype02,
*sequential_3/dense_7/MatMul/ReadVariableOpн
sequential_3/dense_7/MatMulMatMul(sequential_3/dropout_3/Identity:output:02sequential_3/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
sequential_3/dense_7/MatMul╦
+sequential_3/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_3_dense_7_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02-
+sequential_3/dense_7/BiasAdd/ReadVariableOpН
sequential_3/dense_7/BiasAddBiasAdd%sequential_3/dense_7/MatMul:product:03sequential_3/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
sequential_3/dense_7/BiasAddг
"sequential_3/activation_21/SoftmaxSoftmax%sequential_3/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:         d2$
"sequential_3/activation_21/Softmax▓
IdentityIdentity,sequential_3/activation_21/Softmax:softmax:0.^sequential_3/conv2d_10/BiasAdd/ReadVariableOp-^sequential_3/conv2d_10/Conv2D/ReadVariableOp.^sequential_3/conv2d_11/BiasAdd/ReadVariableOp-^sequential_3/conv2d_11/Conv2D/ReadVariableOp.^sequential_3/conv2d_12/BiasAdd/ReadVariableOp-^sequential_3/conv2d_12/Conv2D/ReadVariableOp.^sequential_3/conv2d_13/BiasAdd/ReadVariableOp-^sequential_3/conv2d_13/Conv2D/ReadVariableOp,^sequential_3/dense_6/BiasAdd/ReadVariableOp+^sequential_3/dense_6/MatMul/ReadVariableOp,^sequential_3/dense_7/BiasAdd/ReadVariableOp+^sequential_3/dense_7/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         ╚╚::::::::::::2^
-sequential_3/conv2d_10/BiasAdd/ReadVariableOp-sequential_3/conv2d_10/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_10/Conv2D/ReadVariableOp,sequential_3/conv2d_10/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_11/BiasAdd/ReadVariableOp-sequential_3/conv2d_11/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_11/Conv2D/ReadVariableOp,sequential_3/conv2d_11/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_12/BiasAdd/ReadVariableOp-sequential_3/conv2d_12/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_12/Conv2D/ReadVariableOp,sequential_3/conv2d_12/Conv2D/ReadVariableOp2^
-sequential_3/conv2d_13/BiasAdd/ReadVariableOp-sequential_3/conv2d_13/BiasAdd/ReadVariableOp2\
,sequential_3/conv2d_13/Conv2D/ReadVariableOp,sequential_3/conv2d_13/Conv2D/ReadVariableOp2Z
+sequential_3/dense_6/BiasAdd/ReadVariableOp+sequential_3/dense_6/BiasAdd/ReadVariableOp2X
*sequential_3/dense_6/MatMul/ReadVariableOp*sequential_3/dense_6/MatMul/ReadVariableOp2Z
+sequential_3/dense_7/BiasAdd/ReadVariableOp+sequential_3/dense_7/BiasAdd/ReadVariableOp2X
*sequential_3/dense_7/MatMul/ReadVariableOp*sequential_3/dense_7/MatMul/ReadVariableOp:b ^
1
_output_shapes
:         ╚╚
)
_user_specified_nameconv2d_10_input
Г
L
0__inference_max_pooling2d_12_layer_call_fn_92639

inputs
identityВ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_926332
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
лF
б
G__inference_sequential_3_layer_call_and_return_conditional_losses_93287

inputs,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource,
(conv2d_11_conv2d_readvariableop_resource-
)conv2d_11_biasadd_readvariableop_resource,
(conv2d_12_conv2d_readvariableop_resource-
)conv2d_12_biasadd_readvariableop_resource,
(conv2d_13_conv2d_readvariableop_resource-
)conv2d_13_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource
identityѕб conv2d_10/BiasAdd/ReadVariableOpбconv2d_10/Conv2D/ReadVariableOpб conv2d_11/BiasAdd/ReadVariableOpбconv2d_11/Conv2D/ReadVariableOpб conv2d_12/BiasAdd/ReadVariableOpбconv2d_12/Conv2D/ReadVariableOpб conv2d_13/BiasAdd/ReadVariableOpбconv2d_13/Conv2D/ReadVariableOpбdense_6/BiasAdd/ReadVariableOpбdense_6/MatMul/ReadVariableOpбdense_7/BiasAdd/ReadVariableOpбdense_7/MatMul/ReadVariableOp│
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_10/Conv2D/ReadVariableOp─
conv2d_10/Conv2DConv2Dinputs'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         кк *
paddingVALID*
strides
2
conv2d_10/Conv2Dф
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp▓
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         кк 2
conv2d_10/BiasAddѕ
activation_16/ReluReluconv2d_10/BiasAdd:output:0*
T0*1
_output_shapes
:         кк 2
activation_16/Relu╬
max_pooling2d_10/MaxPoolMaxPool activation_16/Relu:activations:0*/
_output_shapes
:         cc *
ksize
*
paddingVALID*
strides
2
max_pooling2d_10/MaxPool│
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_11/Conv2D/ReadVariableOpП
conv2d_11/Conv2DConv2D!max_pooling2d_10/MaxPool:output:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         aa@*
paddingVALID*
strides
2
conv2d_11/Conv2Dф
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp░
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         aa@2
conv2d_11/BiasAddє
activation_17/ReluReluconv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:         aa@2
activation_17/Relu╬
max_pooling2d_11/MaxPoolMaxPool activation_17/Relu:activations:0*/
_output_shapes
:         00@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_11/MaxPool┤
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02!
conv2d_12/Conv2D/ReadVariableOpя
conv2d_12/Conv2DConv2D!max_pooling2d_11/MaxPool:output:0'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ..ђ*
paddingVALID*
strides
2
conv2d_12/Conv2DФ
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp▒
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ..ђ2
conv2d_12/BiasAddЄ
activation_18/ReluReluconv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:         ..ђ2
activation_18/Relu¤
max_pooling2d_12/MaxPoolMaxPool activation_18/Relu:activations:0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_12/MaxPoolх
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_13/Conv2D/ReadVariableOpя
conv2d_13/Conv2DConv2D!max_pooling2d_12/MaxPool:output:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingVALID*
strides
2
conv2d_13/Conv2DФ
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp▒
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_13/BiasAddЄ
activation_19/ReluReluconv2d_13/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
activation_19/Relu¤
max_pooling2d_13/MaxPoolMaxPool activation_19/Relu:activations:0*0
_output_shapes
:         

ђ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_13/MaxPools
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"     d  2
flatten_3/Constб
flatten_3/ReshapeReshape!max_pooling2d_13/MaxPool:output:0flatten_3/Const:output:0*
T0*)
_output_shapes
:         ђ╚2
flatten_3/Reshapeе
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*!
_output_shapes
:ђ╚ђ*
dtype02
dense_6/MatMul/ReadVariableOpа
dense_6/MatMulMatMulflatten_3/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_6/MatMulЦ
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02 
dense_6/BiasAdd/ReadVariableOpб
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_6/BiasAdd}
activation_20/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
activation_20/ReluЅ
dropout_3/IdentityIdentity activation_20/Relu:activations:0*
T0*(
_output_shapes
:         ђ2
dropout_3/Identityд
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	ђd*
dtype02
dense_7/MatMul/ReadVariableOpа
dense_7/MatMulMatMuldropout_3/Identity:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
dense_7/MatMulц
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02 
dense_7/BiasAdd/ReadVariableOpА
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2
dense_7/BiasAddЁ
activation_21/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:         d2
activation_21/SoftmaxЅ
IdentityIdentityactivation_21/Softmax:softmax:0!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         ╚╚::::::::::::2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:Y U
1
_output_shapes
:         ╚╚
 
_user_specified_nameinputs
Г
L
0__inference_max_pooling2d_11_layer_call_fn_92627

inputs
identityВ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_926212
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
о
d
H__inference_activation_17_layer_call_and_return_conditional_losses_93398

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:         aa@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         aa@2

Identity"
identityIdentity:output:0*.
_input_shapes
:         aa@:W S
/
_output_shapes
:         aa@
 
_user_specified_nameinputs
в	
П
D__inference_conv2d_11_layer_call_and_return_conditional_losses_93384

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         aa@*
paddingVALID*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         aa@2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         aa@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         cc ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         cc 
 
_user_specified_nameinputs
ќ
E
)__inference_dropout_3_layer_call_fn_93528

inputs
identity├
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_928852
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
н	
А
,__inference_sequential_3_layer_call_fn_93060
conv2d_10_input
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

unknown_10
identityѕбStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallconv2d_10_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_930332
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         ╚╚::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:         ╚╚
)
_user_specified_nameconv2d_10_input
║
I
-__inference_activation_17_layer_call_fn_93403

inputs
identity╬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         aa@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation_17_layer_call_and_return_conditional_losses_927262
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         aa@2

Identity"
identityIdentity:output:0*.
_input_shapes
:         aa@:W S
/
_output_shapes
:         aa@
 
_user_specified_nameinputs
Й
d
H__inference_activation_21_layer_call_and_return_conditional_losses_92929

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:         d2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*&
_input_shapes
:         d:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
џ
I
-__inference_activation_21_layer_call_fn_93557

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation_21_layer_call_and_return_conditional_losses_929292
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*&
_input_shapes
:         d:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs
└
`
D__inference_flatten_3_layer_call_and_return_conditional_losses_92821

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"     d  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         ђ╚2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         ђ╚2

Identity"
identityIdentity:output:0*/
_input_shapes
:         

ђ:X T
0
_output_shapes
:         

ђ
 
_user_specified_nameinputs
Џ	
█
B__inference_dense_6_layer_call_and_return_conditional_losses_92839

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpљ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:ђ╚ђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddќ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*0
_input_shapes
:         ђ╚::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:         ђ╚
 
_user_specified_nameinputs
└
`
D__inference_flatten_3_layer_call_and_return_conditional_losses_93467

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"     d  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         ђ╚2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         ђ╚2

Identity"
identityIdentity:output:0*/
_input_shapes
:         

ђ:X T
0
_output_shapes
:         

ђ
 
_user_specified_nameinputs
я
d
H__inference_activation_16_layer_call_and_return_conditional_losses_93369

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:         кк 2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:         кк 2

Identity"
identityIdentity:output:0*0
_input_shapes
:         кк :Y U
1
_output_shapes
:         кк 
 
_user_specified_nameinputs
ё
~
)__inference_conv2d_10_layer_call_fn_93364

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         кк *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_926652
StatefulPartitionedCallў
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:         кк 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:         ╚╚::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ╚╚
 
_user_specified_nameinputs
Ѕ
c
D__inference_dropout_3_layer_call_and_return_conditional_losses_93513

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ђ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeх
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђ2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
╣	
ў
,__inference_sequential_3_layer_call_fn_93345

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

unknown_10
identityѕбStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         d*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_931082
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*`
_input_shapesO
M:         ╚╚::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ╚╚
 
_user_specified_nameinputs
┬
I
-__inference_activation_16_layer_call_fn_93374

inputs
identityл
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         кк * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_activation_16_layer_call_and_return_conditional_losses_926862
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         кк 2

Identity"
identityIdentity:output:0*0
_input_shapes
:         кк :Y U
1
_output_shapes
:         кк 
 
_user_specified_nameinputs
■
~
)__inference_conv2d_12_layer_call_fn_93422

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ..ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv2d_12_layer_call_and_return_conditional_losses_927452
StatefulPartitionedCallЌ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ..ђ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         00@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         00@
 
_user_specified_nameinputs
╦
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_92885

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ђ
g
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_92609

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Й
d
H__inference_activation_21_layer_call_and_return_conditional_losses_93552

inputs
identityW
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:         d2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:         d2

Identity"
identityIdentity:output:0*&
_input_shapes
:         d:O K
'
_output_shapes
:         d
 
_user_specified_nameinputs"▒L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╩
serving_defaultХ
U
conv2d_10_inputB
!serving_default_conv2d_10_input:0         ╚╚A
activation_210
StatefulPartitionedCall:0         dtensorflow/serving/predict:├▄
╔k
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer-14
layer-15
layer_with_weights-5
layer-16
layer-17
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
З__call__
ш_default_save_signature
+Ш&call_and_return_all_conditional_losses"ьf
_tf_keras_sequential╬f{"class_name": "Sequential", "name": "sequential_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 200, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_10_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 200, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_16", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_10", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_17", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_11", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_18", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_12", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_19", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_13", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_21", "trainable": true, "dtype": "float32", "activation": "softmax"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200, 200, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 200, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_10_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 200, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_16", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_10", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_17", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_11", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_18", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_12", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_19", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_13", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_21", "trainable": true, "dtype": "float32", "activation": "softmax"}}]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
■


kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
э__call__
+Э&call_and_return_all_conditional_losses"О	
_tf_keras_layerй	{"class_name": "Conv2D", "name": "conv2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 200, 3]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_10", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 200, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200, 200, 3]}}
┘
regularization_losses
 trainable_variables
!	variables
"	keras_api
щ__call__
+Щ&call_and_return_all_conditional_losses"╚
_tf_keras_layer«{"class_name": "Activation", "name": "activation_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_16", "trainable": true, "dtype": "float32", "activation": "relu"}}
Ѓ
#regularization_losses
$trainable_variables
%	variables
&	keras_api
ч__call__
+Ч&call_and_return_all_conditional_losses"Ы
_tf_keras_layerп{"class_name": "MaxPooling2D", "name": "max_pooling2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_10", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
щ	

'kernel
(bias
)regularization_losses
*trainable_variables
+	variables
,	keras_api
§__call__
+■&call_and_return_all_conditional_losses"м
_tf_keras_layerИ{"class_name": "Conv2D", "name": "conv2d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 99, 99, 32]}}
┘
-regularization_losses
.trainable_variables
/	variables
0	keras_api
 __call__
+ђ&call_and_return_all_conditional_losses"╚
_tf_keras_layer«{"class_name": "Activation", "name": "activation_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_17", "trainable": true, "dtype": "float32", "activation": "relu"}}
Ѓ
1regularization_losses
2trainable_variables
3	variables
4	keras_api
Ђ__call__
+ѓ&call_and_return_all_conditional_losses"Ы
_tf_keras_layerп{"class_name": "MaxPooling2D", "name": "max_pooling2d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_11", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Щ	

5kernel
6bias
7regularization_losses
8trainable_variables
9	variables
:	keras_api
Ѓ__call__
+ё&call_and_return_all_conditional_losses"М
_tf_keras_layer╣{"class_name": "Conv2D", "name": "conv2d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_12", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 48, 64]}}
┘
;regularization_losses
<trainable_variables
=	variables
>	keras_api
Ё__call__
+є&call_and_return_all_conditional_losses"╚
_tf_keras_layer«{"class_name": "Activation", "name": "activation_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_18", "trainable": true, "dtype": "float32", "activation": "relu"}}
Ѓ
?regularization_losses
@trainable_variables
A	variables
B	keras_api
Є__call__
+ѕ&call_and_return_all_conditional_losses"Ы
_tf_keras_layerп{"class_name": "MaxPooling2D", "name": "max_pooling2d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_12", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ч	

Ckernel
Dbias
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
Ѕ__call__
+і&call_and_return_all_conditional_losses"Н
_tf_keras_layer╗{"class_name": "Conv2D", "name": "conv2d_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 23, 23, 128]}}
┘
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
І__call__
+ї&call_and_return_all_conditional_losses"╚
_tf_keras_layer«{"class_name": "Activation", "name": "activation_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_19", "trainable": true, "dtype": "float32", "activation": "relu"}}
Ѓ
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
Ї__call__
+ј&call_and_return_all_conditional_losses"Ы
_tf_keras_layerп{"class_name": "MaxPooling2D", "name": "max_pooling2d_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_13", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
У
Qregularization_losses
Rtrainable_variables
S	variables
T	keras_api
Ј__call__
+љ&call_and_return_all_conditional_losses"О
_tf_keras_layerй{"class_name": "Flatten", "name": "flatten_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ч

Ukernel
Vbias
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
Љ__call__
+њ&call_and_return_all_conditional_losses"н
_tf_keras_layer║{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 25600}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 25600]}}
┘
[regularization_losses
\trainable_variables
]	variables
^	keras_api
Њ__call__
+ћ&call_and_return_all_conditional_losses"╚
_tf_keras_layer«{"class_name": "Activation", "name": "activation_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_20", "trainable": true, "dtype": "float32", "activation": "relu"}}
у
_regularization_losses
`trainable_variables
a	variables
b	keras_api
Ћ__call__
+ќ&call_and_return_all_conditional_losses"о
_tf_keras_layer╝{"class_name": "Dropout", "name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
э

ckernel
dbias
eregularization_losses
ftrainable_variables
g	variables
h	keras_api
Ќ__call__
+ў&call_and_return_all_conditional_losses"л
_tf_keras_layerХ{"class_name": "Dense", "name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 100, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
▄
iregularization_losses
jtrainable_variables
k	variables
l	keras_api
Ў__call__
+џ&call_and_return_all_conditional_losses"╦
_tf_keras_layer▒{"class_name": "Activation", "name": "activation_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_21", "trainable": true, "dtype": "float32", "activation": "softmax"}}
├
miter

nbeta_1

obeta_2
	pdecay
qlearning_ratem▄mП'mя(m▀5mЯ6mрCmРDmсUmСVmтcmТdmуvУvж'vЖ(vв5vВ6vьCvЬDv№Uv­VvыcvЫdvз"
	optimizer
 "
trackable_list_wrapper
v
0
1
'2
(3
54
65
C6
D7
U8
V9
c10
d11"
trackable_list_wrapper
v
0
1
'2
(3
54
65
C6
D7
U8
V9
c10
d11"
trackable_list_wrapper
╬
rlayer_metrics
snon_trainable_variables
regularization_losses
trainable_variables
	variables
tmetrics

ulayers
vlayer_regularization_losses
З__call__
ш_default_save_signature
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
-
Џserving_default"
signature_map
*:( 2conv2d_10/kernel
: 2conv2d_10/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
░
wlayer_metrics
xnon_trainable_variables
regularization_losses
trainable_variables
	variables
ymetrics

zlayers
{layer_regularization_losses
э__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▒
|layer_metrics
}non_trainable_variables
regularization_losses
 trainable_variables
!	variables
~metrics

layers
 ђlayer_regularization_losses
щ__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Ђlayer_metrics
ѓnon_trainable_variables
#regularization_losses
$trainable_variables
%	variables
Ѓmetrics
ёlayers
 Ёlayer_regularization_losses
ч__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv2d_11/kernel
:@2conv2d_11/bias
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
х
єlayer_metrics
Єnon_trainable_variables
)regularization_losses
*trainable_variables
+	variables
ѕmetrics
Ѕlayers
 іlayer_regularization_losses
§__call__
+■&call_and_return_all_conditional_losses
'■"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Іlayer_metrics
їnon_trainable_variables
-regularization_losses
.trainable_variables
/	variables
Їmetrics
јlayers
 Јlayer_regularization_losses
 __call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
љlayer_metrics
Љnon_trainable_variables
1regularization_losses
2trainable_variables
3	variables
њmetrics
Њlayers
 ћlayer_regularization_losses
Ђ__call__
+ѓ&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses"
_generic_user_object
+:)@ђ2conv2d_12/kernel
:ђ2conv2d_12/bias
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
х
Ћlayer_metrics
ќnon_trainable_variables
7regularization_losses
8trainable_variables
9	variables
Ќmetrics
ўlayers
 Ўlayer_regularization_losses
Ѓ__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
џlayer_metrics
Џnon_trainable_variables
;regularization_losses
<trainable_variables
=	variables
юmetrics
Юlayers
 ъlayer_regularization_losses
Ё__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Ъlayer_metrics
аnon_trainable_variables
?regularization_losses
@trainable_variables
A	variables
Аmetrics
бlayers
 Бlayer_regularization_losses
Є__call__
+ѕ&call_and_return_all_conditional_losses
'ѕ"call_and_return_conditional_losses"
_generic_user_object
,:*ђђ2conv2d_13/kernel
:ђ2conv2d_13/bias
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
х
цlayer_metrics
Цnon_trainable_variables
Eregularization_losses
Ftrainable_variables
G	variables
дmetrics
Дlayers
 еlayer_regularization_losses
Ѕ__call__
+і&call_and_return_all_conditional_losses
'і"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Еlayer_metrics
фnon_trainable_variables
Iregularization_losses
Jtrainable_variables
K	variables
Фmetrics
гlayers
 Гlayer_regularization_losses
І__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
«layer_metrics
»non_trainable_variables
Mregularization_losses
Ntrainable_variables
O	variables
░metrics
▒layers
 ▓layer_regularization_losses
Ї__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
│layer_metrics
┤non_trainable_variables
Qregularization_losses
Rtrainable_variables
S	variables
хmetrics
Хlayers
 иlayer_regularization_losses
Ј__call__
+љ&call_and_return_all_conditional_losses
'љ"call_and_return_conditional_losses"
_generic_user_object
#:!ђ╚ђ2dense_6/kernel
:ђ2dense_6/bias
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
х
Иlayer_metrics
╣non_trainable_variables
Wregularization_losses
Xtrainable_variables
Y	variables
║metrics
╗layers
 ╝layer_regularization_losses
Љ__call__
+њ&call_and_return_all_conditional_losses
'њ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
йlayer_metrics
Йnon_trainable_variables
[regularization_losses
\trainable_variables
]	variables
┐metrics
└layers
 ┴layer_regularization_losses
Њ__call__
+ћ&call_and_return_all_conditional_losses
'ћ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
┬layer_metrics
├non_trainable_variables
_regularization_losses
`trainable_variables
a	variables
─metrics
┼layers
 кlayer_regularization_losses
Ћ__call__
+ќ&call_and_return_all_conditional_losses
'ќ"call_and_return_conditional_losses"
_generic_user_object
!:	ђd2dense_7/kernel
:d2dense_7/bias
 "
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
х
Кlayer_metrics
╚non_trainable_variables
eregularization_losses
ftrainable_variables
g	variables
╔metrics
╩layers
 ╦layer_regularization_losses
Ќ__call__
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
╠layer_metrics
═non_trainable_variables
iregularization_losses
jtrainable_variables
k	variables
╬metrics
¤layers
 лlayer_regularization_losses
Ў__call__
+џ&call_and_return_all_conditional_losses
'џ"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
Л0
м1"
trackable_list_wrapper
д
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
12
13
14
15
16
17"
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
┐

Мtotal

нcount
Н	variables
о	keras_api"ё
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
І

Оtotal

пcount
┘
_fn_kwargs
┌	variables
█	keras_api"┐
_tf_keras_metricц{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:  (2total
:  (2count
0
М0
н1"
trackable_list_wrapper
.
Н	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
О0
п1"
trackable_list_wrapper
.
┌	variables"
_generic_user_object
/:- 2Adam/conv2d_10/kernel/m
!: 2Adam/conv2d_10/bias/m
/:- @2Adam/conv2d_11/kernel/m
!:@2Adam/conv2d_11/bias/m
0:.@ђ2Adam/conv2d_12/kernel/m
": ђ2Adam/conv2d_12/bias/m
1:/ђђ2Adam/conv2d_13/kernel/m
": ђ2Adam/conv2d_13/bias/m
(:&ђ╚ђ2Adam/dense_6/kernel/m
 :ђ2Adam/dense_6/bias/m
&:$	ђd2Adam/dense_7/kernel/m
:d2Adam/dense_7/bias/m
/:- 2Adam/conv2d_10/kernel/v
!: 2Adam/conv2d_10/bias/v
/:- @2Adam/conv2d_11/kernel/v
!:@2Adam/conv2d_11/bias/v
0:.@ђ2Adam/conv2d_12/kernel/v
": ђ2Adam/conv2d_12/bias/v
1:/ђђ2Adam/conv2d_13/kernel/v
": ђ2Adam/conv2d_13/bias/v
(:&ђ╚ђ2Adam/dense_6/kernel/v
 :ђ2Adam/dense_6/bias/v
&:$	ђd2Adam/dense_7/kernel/v
:d2Adam/dense_7/bias/v
■2ч
,__inference_sequential_3_layer_call_fn_93135
,__inference_sequential_3_layer_call_fn_93345
,__inference_sequential_3_layer_call_fn_93316
,__inference_sequential_3_layer_call_fn_93060└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
­2ь
 __inference__wrapped_model_92603╚
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *8б5
3і0
conv2d_10_input         ╚╚
Ж2у
G__inference_sequential_3_layer_call_and_return_conditional_losses_93287
G__inference_sequential_3_layer_call_and_return_conditional_losses_92984
G__inference_sequential_3_layer_call_and_return_conditional_losses_93234
G__inference_sequential_3_layer_call_and_return_conditional_losses_92938└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
М2л
)__inference_conv2d_10_layer_call_fn_93364б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_conv2d_10_layer_call_and_return_conditional_losses_93355б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
О2н
-__inference_activation_16_layer_call_fn_93374б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ы2№
H__inference_activation_16_layer_call_and_return_conditional_losses_93369б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ў2Ћ
0__inference_max_pooling2d_10_layer_call_fn_92615Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
│2░
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_92609Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
М2л
)__inference_conv2d_11_layer_call_fn_93393б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_conv2d_11_layer_call_and_return_conditional_losses_93384б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
О2н
-__inference_activation_17_layer_call_fn_93403б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ы2№
H__inference_activation_17_layer_call_and_return_conditional_losses_93398б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ў2Ћ
0__inference_max_pooling2d_11_layer_call_fn_92627Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
│2░
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_92621Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
М2л
)__inference_conv2d_12_layer_call_fn_93422б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_conv2d_12_layer_call_and_return_conditional_losses_93413б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
О2н
-__inference_activation_18_layer_call_fn_93432б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ы2№
H__inference_activation_18_layer_call_and_return_conditional_losses_93427б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ў2Ћ
0__inference_max_pooling2d_12_layer_call_fn_92639Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
│2░
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_92633Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
М2л
)__inference_conv2d_13_layer_call_fn_93451б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_conv2d_13_layer_call_and_return_conditional_losses_93442б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
О2н
-__inference_activation_19_layer_call_fn_93461б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ы2№
H__inference_activation_19_layer_call_and_return_conditional_losses_93456б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ў2Ћ
0__inference_max_pooling2d_13_layer_call_fn_92651Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
│2░
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_92645Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
М2л
)__inference_flatten_3_layer_call_fn_93472б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_flatten_3_layer_call_and_return_conditional_losses_93467б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_dense_6_layer_call_fn_93491б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
В2ж
B__inference_dense_6_layer_call_and_return_conditional_losses_93482б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
О2н
-__inference_activation_20_layer_call_fn_93501б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ы2№
H__inference_activation_20_layer_call_and_return_conditional_losses_93496б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
љ2Ї
)__inference_dropout_3_layer_call_fn_93523
)__inference_dropout_3_layer_call_fn_93528┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
к2├
D__inference_dropout_3_layer_call_and_return_conditional_losses_93513
D__inference_dropout_3_layer_call_and_return_conditional_losses_93518┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Л2╬
'__inference_dense_7_layer_call_fn_93547б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
В2ж
B__inference_dense_7_layer_call_and_return_conditional_losses_93538б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
О2н
-__inference_activation_21_layer_call_fn_93557б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ы2№
H__inference_activation_21_layer_call_and_return_conditional_losses_93552б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
мB¤
#__inference_signature_wrapper_93174conv2d_10_input"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 Х
 __inference__wrapped_model_92603Љ'(56CDUVcdBб?
8б5
3і0
conv2d_10_input         ╚╚
ф "=ф:
8
activation_21'і$
activation_21         dИ
H__inference_activation_16_layer_call_and_return_conditional_losses_93369l9б6
/б,
*і'
inputs         кк 
ф "/б,
%і"
0         кк 
џ љ
-__inference_activation_16_layer_call_fn_93374_9б6
/б,
*і'
inputs         кк 
ф ""і         кк ┤
H__inference_activation_17_layer_call_and_return_conditional_losses_93398h7б4
-б*
(і%
inputs         aa@
ф "-б*
#і 
0         aa@
џ ї
-__inference_activation_17_layer_call_fn_93403[7б4
-б*
(і%
inputs         aa@
ф " і         aa@Х
H__inference_activation_18_layer_call_and_return_conditional_losses_93427j8б5
.б+
)і&
inputs         ..ђ
ф ".б+
$і!
0         ..ђ
џ ј
-__inference_activation_18_layer_call_fn_93432]8б5
.б+
)і&
inputs         ..ђ
ф "!і         ..ђХ
H__inference_activation_19_layer_call_and_return_conditional_losses_93456j8б5
.б+
)і&
inputs         ђ
ф ".б+
$і!
0         ђ
џ ј
-__inference_activation_19_layer_call_fn_93461]8б5
.б+
)і&
inputs         ђ
ф "!і         ђд
H__inference_activation_20_layer_call_and_return_conditional_losses_93496Z0б-
&б#
!і
inputs         ђ
ф "&б#
і
0         ђ
џ ~
-__inference_activation_20_layer_call_fn_93501M0б-
&б#
!і
inputs         ђ
ф "і         ђц
H__inference_activation_21_layer_call_and_return_conditional_losses_93552X/б,
%б"
 і
inputs         d
ф "%б"
і
0         d
џ |
-__inference_activation_21_layer_call_fn_93557K/б,
%б"
 і
inputs         d
ф "і         dИ
D__inference_conv2d_10_layer_call_and_return_conditional_losses_93355p9б6
/б,
*і'
inputs         ╚╚
ф "/б,
%і"
0         кк 
џ љ
)__inference_conv2d_10_layer_call_fn_93364c9б6
/б,
*і'
inputs         ╚╚
ф ""і         кк ┤
D__inference_conv2d_11_layer_call_and_return_conditional_losses_93384l'(7б4
-б*
(і%
inputs         cc 
ф "-б*
#і 
0         aa@
џ ї
)__inference_conv2d_11_layer_call_fn_93393_'(7б4
-б*
(і%
inputs         cc 
ф " і         aa@х
D__inference_conv2d_12_layer_call_and_return_conditional_losses_93413m567б4
-б*
(і%
inputs         00@
ф ".б+
$і!
0         ..ђ
џ Ї
)__inference_conv2d_12_layer_call_fn_93422`567б4
-б*
(і%
inputs         00@
ф "!і         ..ђХ
D__inference_conv2d_13_layer_call_and_return_conditional_losses_93442nCD8б5
.б+
)і&
inputs         ђ
ф ".б+
$і!
0         ђ
џ ј
)__inference_conv2d_13_layer_call_fn_93451aCD8б5
.б+
)і&
inputs         ђ
ф "!і         ђЦ
B__inference_dense_6_layer_call_and_return_conditional_losses_93482_UV1б.
'б$
"і
inputs         ђ╚
ф "&б#
і
0         ђ
џ }
'__inference_dense_6_layer_call_fn_93491RUV1б.
'б$
"і
inputs         ђ╚
ф "і         ђБ
B__inference_dense_7_layer_call_and_return_conditional_losses_93538]cd0б-
&б#
!і
inputs         ђ
ф "%б"
і
0         d
џ {
'__inference_dense_7_layer_call_fn_93547Pcd0б-
&б#
!і
inputs         ђ
ф "і         dд
D__inference_dropout_3_layer_call_and_return_conditional_losses_93513^4б1
*б'
!і
inputs         ђ
p
ф "&б#
і
0         ђ
џ д
D__inference_dropout_3_layer_call_and_return_conditional_losses_93518^4б1
*б'
!і
inputs         ђ
p 
ф "&б#
і
0         ђ
џ ~
)__inference_dropout_3_layer_call_fn_93523Q4б1
*б'
!і
inputs         ђ
p
ф "і         ђ~
)__inference_dropout_3_layer_call_fn_93528Q4б1
*б'
!і
inputs         ђ
p 
ф "і         ђФ
D__inference_flatten_3_layer_call_and_return_conditional_losses_93467c8б5
.б+
)і&
inputs         

ђ
ф "'б$
і
0         ђ╚
џ Ѓ
)__inference_flatten_3_layer_call_fn_93472V8б5
.б+
)і&
inputs         

ђ
ф "і         ђ╚Ь
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_92609ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ к
0__inference_max_pooling2d_10_layer_call_fn_92615ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    Ь
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_92621ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ к
0__inference_max_pooling2d_11_layer_call_fn_92627ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    Ь
K__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_92633ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ к
0__inference_max_pooling2d_12_layer_call_fn_92639ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    Ь
K__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_92645ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ к
0__inference_max_pooling2d_13_layer_call_fn_92651ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    ═
G__inference_sequential_3_layer_call_and_return_conditional_losses_92938Ђ'(56CDUVcdJбG
@б=
3і0
conv2d_10_input         ╚╚
p

 
ф "%б"
і
0         d
џ ═
G__inference_sequential_3_layer_call_and_return_conditional_losses_92984Ђ'(56CDUVcdJбG
@б=
3і0
conv2d_10_input         ╚╚
p 

 
ф "%б"
і
0         d
џ ├
G__inference_sequential_3_layer_call_and_return_conditional_losses_93234x'(56CDUVcdAб>
7б4
*і'
inputs         ╚╚
p

 
ф "%б"
і
0         d
џ ├
G__inference_sequential_3_layer_call_and_return_conditional_losses_93287x'(56CDUVcdAб>
7б4
*і'
inputs         ╚╚
p 

 
ф "%б"
і
0         d
џ ц
,__inference_sequential_3_layer_call_fn_93060t'(56CDUVcdJбG
@б=
3і0
conv2d_10_input         ╚╚
p

 
ф "і         dц
,__inference_sequential_3_layer_call_fn_93135t'(56CDUVcdJбG
@б=
3і0
conv2d_10_input         ╚╚
p 

 
ф "і         dЏ
,__inference_sequential_3_layer_call_fn_93316k'(56CDUVcdAб>
7б4
*і'
inputs         ╚╚
p

 
ф "і         dЏ
,__inference_sequential_3_layer_call_fn_93345k'(56CDUVcdAб>
7б4
*і'
inputs         ╚╚
p 

 
ф "і         d╠
#__inference_signature_wrapper_93174ц'(56CDUVcdUбR
б 
KфH
F
conv2d_10_input3і0
conv2d_10_input         ╚╚"=ф:
8
activation_21'і$
activation_21         d