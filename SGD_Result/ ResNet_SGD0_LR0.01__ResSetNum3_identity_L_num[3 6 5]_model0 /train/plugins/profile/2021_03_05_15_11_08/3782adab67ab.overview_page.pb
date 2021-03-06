?	?m??_?~@?m??_?~@!?m??_?~@	?Ϭ?????Ϭ????!?Ϭ????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?m??_?~@z?}??v@1&??>^@AU????,??I???^z$@Y[	?%q6@*	S㥛$y?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorO???*?$@!?k?avS@)O???*?$@1?k?avS@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?VC??	@!?;t? ?7@)?VC??	@1?;t? ?7@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?A?V??	@!o?z???7@)S?u8?J??1`?"??z??:Preprocessing2F
Iterator::Modelk?K??	@!gp?L?7@)?_?5?!z?11?݃???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap
???$@!???,?S@)5?+-#?n?1?&?W????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 73.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?Ϭ????I?@2!??R@Q[J?#Wq8@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	z?}??v@z?}??v@!z?}??v@      ??!       "	&??>^@&??>^@!&??>^@*      ??!       2	U????,??U????,??!U????,??:	???^z$@???^z$@!???^z$@B      ??!       J	[	?%q6@[	?%q6@![	?%q6@R      ??!       Z	[	?%q6@[	?%q6@![	?%q6@b      ??!       JGPUY?Ϭ????b q?@2!??R@y[J?#Wq8@?"7
model_4/conv2d_206/Conv2DConv2D??܊?M??!??܊?M??0"l
Bgradient_tape/model_4/batch_normalization_206/FusedBatchNormGradV3FusedBatchNormGradV3<.??$??!????$p??"f
;gradient_tape/model_4/conv2d_210/Conv2D/Conv2DBackpropInputConv2DBackpropInput.6?-?e??!??BN?|??0"V
0model_4/batch_normalization_206/FusedBatchNormV3FusedBatchNormV3?|?eև?!]S???w??"7
model_4/conv2d_206/BiasAddBiasAddx?g??}??!LM?8RG??"K
-gradient_tape/model_4/activation_190/ReluGradReluGrad?%(g???!J??_??"f
;gradient_tape/model_4/conv2d_207/Conv2D/Conv2DBackpropInputConv2DBackpropInput??wWQ???!??HI???0"7
model_4/conv2d_215/Conv2DConv2D????ߤ??!:??7??0"7
model_4/conv2d_208/Conv2DConv2D???Ő??!~gd?????0"f
;gradient_tape/model_4/conv2d_208/Conv2D/Conv2DBackpropInputConv2DBackpropInputi?bYRk??!???????0Q      Y@Y?va?????a&z6 ??X@q???@@y?n?p?"?

both?Your program is POTENTIALLY input-bound because 73.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?33.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 