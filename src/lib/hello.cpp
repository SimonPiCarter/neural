#include "hello.h"

std::string test::hello(std::string const & name_p)
{
	return std::string("Hello "+name_p+"!");
}