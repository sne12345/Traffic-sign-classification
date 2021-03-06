?	n3??6?@n3??6?@!n3??6?@	?^?????^????!?^????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6n3??6?@??o?(?x@1┹???m@A}?|?.P??IDkE???@YSͬ????*	/?$?l?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator???{?@!h鍌??X@)???{?@1h鍌??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??aMe??!??iԵ???)??aMe??1??iԵ???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?2?g???!???M????)0???hȘ?1? ??-???:Preprocessing2F
Iterator::Model?fh<??!?ईB??)T?^Pz?1??^?#յ?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??W\?@!}lݱ??X@)????
o?1{?|*???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 61.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?^????I&??<?oO@Q w,?ƋB@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??o?(?x@??o?(?x@!??o?(?x@      ??!       "	┹???m@┹???m@!┹???m@*      ??!       2	}?|?.P??}?|?.P??!}?|?.P??:	DkE???@DkE???@!DkE???@B      ??!       J	Sͬ????Sͬ????!Sͬ????R      ??!       Z	Sͬ????Sͬ????!Sͬ????b      ??!       JGPUY?^????b q&??<?oO@y w,?ƋB@?"8
model_18/conv2d_964/Conv2DConv2D???V???!???V???0"i
=gradient_tape/model_18/conv2d_966/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterV`PY???!??7?WL??0"i
=gradient_tape/model_18/conv2d_973/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterǖ????!{???ў??0"i
=gradient_tape/model_18/conv2d_970/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter6g?????!H??????0"i
=gradient_tape/model_18/conv2d_976/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter ??Aԕ?!???9???0"g
<gradient_tape/model_18/conv2d_972/Conv2D/Conv2DBackpropInputConv2DBackpropInput??E[,m??!N%???,??0"g
<gradient_tape/model_18/conv2d_975/Conv2D/Conv2DBackpropInputConv2DBackpropInput??wu?k??!??*????0"g
<gradient_tape/model_18/conv2d_969/Conv2D/Conv2DBackpropInputConv2DBackpropInput#Sk?^g??!?R????0"8
model_18/conv2d_973/Conv2DConv2D?????w??!2?X?A??0"8
model_18/conv2d_970/Conv2DConv2DI&??Ru??!~D??و??0Q      Y@Y?]?/7???a??A#??X@q?<ʖF@y?gZ??vb?"?

both?Your program is POTENTIALLY input-bound because 61.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?44.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 