#version 140
//#define GL_compatibility_profile 1
#extension GL_ARB_compatibility: enable

uniform mat4 modelMat;
uniform mat4 viewMat;
uniform mat4 projMat;

varying vec3 N;
varying vec3 v;

void main(void)  
{
	N = normalize(gl_NormalMatrix * gl_Normal);
	//v = vec3(gl_ModelViewMatrix * gl_Vertex);
	//gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * gl_Vertex; 

	mat4 modelView = viewMat * modelMat;
	mat4 modelViewProjection = projMat * modelView;
	v = vec3(modelView * gl_Vertex);
	gl_Position = modelViewProjection * gl_Vertex;
}