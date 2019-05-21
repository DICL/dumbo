// handlebars.js 커스텀함수 정의

/**
 * {{math index "+" 1}} 결과값 출력
 */
Handlebars.registerHelper('math', function(lvalue, operator, rvalue) {
	lvalue = parseFloat(lvalue);
	rvalue = parseFloat(rvalue);
	return {
        "+": lvalue + rvalue,
        "-": lvalue - rvalue,
        "*": lvalue * rvalue,
        "/": lvalue / rvalue,
        "%": lvalue % rvalue
    }[operator];
});


/**
 * {{substr text 3}}
 */
Handlebars.registerHelper('substr', function(string, num) {
	try {
		var result = string.substr(0,num);
	} catch (e) {
		var result = '';
		console.warn(e)
	}
	return result;
});

/**
 * {{if_text index '==' '0'}} true : false
 */
Handlebars.registerHelper('if_text', function (v1, operator, v2) {
    switch (operator) {
        case '==':
            return (v1 == v2) ;
        case '===':
            return (v1 === v2) ;
        case '!=':
            return (v1 != v2) ;
        case '!==':
            return (v1 !== v2) ;
        case '<':
            return (v1 < v2) ;
        case '<=':
            return (v1 <= v2) ;
        case '>':
            return (v1 > v2) ;
        case '>=':
            return (v1 >= v2) ;
        case '&&':
            return (v1 && v2) ;
        case '||':
            return (v1 || v2) ;
    }
});
/**
 * {{#ifCond index '==' 0}}
 * 	true 
 * {{else}}
 *  false
 * {{/ifCond }}
 */
Handlebars.registerHelper('ifCond', function (v1, operator, v2, options) {
    switch (operator) {
        case '==':
            return (v1 == v2) ? options.fn(this) : options.inverse(this);
        case '===':
            return (v1 === v2) ? options.fn(this) : options.inverse(this);
        case '!=':
            return (v1 != v2) ? options.fn(this) : options.inverse(this);
        case '!==':
            return (v1 !== v2) ? options.fn(this) : options.inverse(this);
        case '<':
            return (v1 < v2) ? options.fn(this) : options.inverse(this);
        case '<=':
            return (v1 <= v2) ? options.fn(this) : options.inverse(this);
        case '>':
            return (v1 > v2) ? options.fn(this) : options.inverse(this);
        case '>=':
            return (v1 >= v2) ? options.fn(this) : options.inverse(this);
        case '&&':
            return (v1 && v2) ? options.fn(this) : options.inverse(this);
        case '||':
            return (v1 || v2) ? options.fn(this) : options.inverse(this);
        default:
            return options.inverse(this);
    }
});

Handlebars.registerHelper('set_log_type',function(value){
	var result = '';
	var length = 5;

	switch (value) {
		case 'command':	
			result = 'CMD';
			break;
		case 'info':	
			result = 'INFO';
			break;
		case 'error':	
			result = 'ERROR';
			break;
		default:
			break;
	}
	
	var str=""
	for(var i=0;i<length-result.length;i++){
	  str=str+"&#160;";
	}
	result=result+str;
	
	return result;
});


/**
 * {{#for 0 ../disk_total_count 1}}
 * {{/for}}
 */
Handlebars.registerHelper('for', function(from, to, incr, block) {
    var accum = '';
    for(var i = from; i < to; i += incr)
        accum += block.fn(i);
    return accum;
});
