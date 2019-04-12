#version 140
#extension GL_ARB_compatibility: enable

//uniform float specIntensity;
//uniform float ambiIntensity;
//uniform float diffIntensity;
//uniform float shininess;
//uniform vec3 lpos;
//uniform float vbo;

uniform vec4 ambient_color;
uniform vec4 diffuse_color;


varying vec3 N;
varying vec3 v;
varying vec4 color;

void main (void)  
{
	//vec4 specular = vec4(1.0,1.0,1.0,1.0) * 1.0;
	//vec4 ambient = ambient_color * 0.38;
	//vec4 diffuse = diffuse_color * 0.64;

	vec4 specular = vec4(1.0,1.0,1.0,1.0) * 0.44;
	vec4 ambient = color * 0.3;
	vec4 diffuse = color * 0.8;

	//vec3 L = normalize(vec3(1.0, 1.0, 0) - v);
	vec3 L = normalize(vec3(1.0, 1.0, -1.0) - v);
	vec3 E = normalize(-1.0*v);
	vec3 R = normalize(-reflect(L,N));

	diffuse = diffuse * max(dot(N,L), 0.0);
	diffuse = clamp(diffuse, 0.0, 1.0);

	specular = specular * pow(max(dot(R,E),0.0),0.3*100);
	specular = clamp(specular, 0.0, 1.0); 

	vec4 final = ambient + diffuse + specular;
    final.w = 0.1;
	gl_FragColor = final;
	//gl_FragColor = ambient + diffuse + specular;
	//gl_FragColor = diffuse;
}
