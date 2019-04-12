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
void main (void)  
{
	vec4 specular = vec4(1.0,1.0,1.0,1.0) * 1.0;
	vec4 ambient = ambient_color * 0.3;
	vec4 diffuse = diffuse_color * 0.50;

	vec3 L = normalize(vec3(-35.0,60.0,-50.0) - v);
	vec3 E = normalize(-1.0*v);
	vec3 R = normalize(-reflect(L,N));

	diffuse = diffuse * max(dot(N,L), 0.0);
	diffuse = clamp(diffuse, 0.0, 1.0);

	specular = specular * pow(max(dot(R,E),0.0),0.5*100);
	specular = clamp(specular, 0.0, 1.0); 

	//gl_FragColor = ambient;
	//gl_FragColor = ambient*0.5 + diffuse*0.5;
	//gl_FragColor = specular;
	gl_FragColor = ambient + diffuse + specular;
}
