?	qǛ?V!?@qǛ?V!?@!qǛ?V!?@	??/A????/A??!??/A??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6qǛ?V!?@???4??t@1/?r?],d@A?@?S????Im????2@Y???/fK??*	??? ?G?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator>[{?)@!?}g?X@)>[{?)@1?}g?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchQ?Hm???!%C?mb+??)Q?Hm???1%C?mb+??:Preprocessing2F
Iterator::Model?~?~?d??!þ<R????)o?UfJ???1y???????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?#d Ϻ?!?=?e(???)?z?p̲??1?W????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??@?M?)@!??ͼX@)1'h??'m?1W?H;(??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 65.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??/A??I???+Q@Q,??+D?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???4??t@???4??t@!???4??t@      ??!       "	/?r?],d@/?r?],d@!/?r?],d@*      ??!       2	?@?S?????@?S????!?@?S????:	m????2@m????2@!m????2@B      ??!       J	???/fK?????/fK??!???/fK??R      ??!       Z	???/fK?????/fK??!???/fK??b      ??!       JGPUY??/A??b q???+Q@y,??+D?@?"7
model_3/conv2d_153/Conv2DConv2D???ۢ?!???ۢ?0"h
<gradient_tape/model_3/conv2d_162/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterˈJqr=??!?ڝ)}??0"h
<gradient_tape/model_3/conv2d_155/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterjʿ?d5??!.J?????0"h
<gradient_tape/model_3/conv2d_159/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?Ɗ?2??!۴l????0"f
;gradient_tape/model_3/conv2d_158/Conv2D/Conv2DBackpropInputConv2DBackpropInputqgi???!??FԲ??0"f
;gradient_tape/model_3/conv2d_161/Conv2D/Conv2DBackpropInputConv2DBackpropInputUU?\???!6??h???0"l
Bgradient_tape/model_3/batch_normalization_153/FusedBatchNormGradV3FusedBatchNormGradV3?S[?????!p??>????"l
Bgradient_tape/model_3/batch_normalization_157/FusedBatchNormGradV3FusedBatchNormGradV30K??L???!#0?	?G??"l
Bgradient_tape/model_3/batch_normalization_163/FusedBatchNormGradV3FusedBatchNormGradV3%;?/????!???܉???"l
Bgradient_tape/model_3/batch_normalization_160/FusedBatchNormGradV3FusedBatchNormGradV3??IM????!??l?????Q      Y@Y???????a?A?g?X@q?֮4l?@y??`???h?"?
both?Your program is POTENTIALLY input-bound because 65.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?31.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 