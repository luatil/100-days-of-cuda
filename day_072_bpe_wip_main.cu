#include <stdio.h>
#include <wchar.h>
#include <locale.h>
#include <string.h>

static int StringLength(const char *String)
{
    return strlen(String);
}

int main()
{
    setlocale(LC_ALL, "");  // Enable locale support

    const char *Bytes = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception.";

    const int Length = StringLength(Bytes);

    int Idx = 0;

    while (Idx < Length)
    {
	// Print CodePoint of first character
	unsigned int CodePoint = 0;
	int CharBytes = 0;
	
	// 1-byte character (ASCII)
	if ((Bytes[Idx] & 0x80) == 0)
	{
	    CodePoint = Bytes[Idx];
	    CharBytes = 1;
	}
	// 2-byte character
	else if ((Bytes[Idx] & 0xE0) == 0xC0)
	{
	    CodePoint = ((Bytes[Idx] & 0x1F) << 6) | (Bytes[Idx + 1] & 0x3F);
	    CharBytes = 2;
	}
	// 3-byte character
	else if ((Bytes[Idx] & 0xF0) == 0xE0)
	{
	    CodePoint = ((Bytes[Idx] & 0x0F) << 12) | ((Bytes[Idx+1] & 0x3F) << 6) | (Bytes[Idx+2] & 0x3F);
	    CharBytes = 3;
	}
	// 4-byte character
	else if ((Bytes[Idx] & 0xF8) == 0xF0)
	{
	    CodePoint = ((Bytes[Idx] & 0x07) << 18) | ((Bytes[Idx+1] & 0x3F) << 12) | ((Bytes[Idx+2] & 0x3F) << 6) | (Bytes[Idx+3] & 0x3F);
	    CharBytes = 4;
	}
	
	printf("[%d] \t %.*s \t U+%04X \t (%u)\n", CharBytes, CharBytes, &Bytes[Idx], CodePoint, CodePoint);
	Idx += CharBytes;
    }
}

