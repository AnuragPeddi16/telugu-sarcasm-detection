// Clean, native implementation using Google's Input Tools API without any external dependencies

export const useGoogleInputTool = (elementId: string, isMounted: boolean) => {
  // Deprecated: No longer needed with the native interceptor below
};

export async function handleTransliterationInput(
  inputValue: string,
  setTransliteratedText: (text: string) => void
) {
  // First update exactly what the user typed so UI doesn't lag
  setTransliteratedText(inputValue);

  // Check if the input ends with a space, newline, or a punctuation mark
  // Punctuation marks: . , ! ? ; : " ' ) ] }
  const triggerRegex = /[\s\n.,!?;:"')\]}]$/;
  
  if (triggerRegex.test(inputValue)) {
    // Find the last word boundaries
    // We split by any whitespace or punctuation to isolate the word
    const lastChar = inputValue.slice(-1);
    const textWithoutTrigger = inputValue.slice(0, -1);
    
    // Split the remaining text by non-alphanumeric to find the last true word token
    const words = textWithoutTrigger.split(/([\s\n.,!?;:"')\]}]+)/);
    const lastWordIdx = words.length - 1;
    const lastWord = words[lastWordIdx];
    
    // Process if it's strictly an English alphabet word
    if (lastWord && /^[a-zA-Z]+$/.test(lastWord)) {
      try {
        // Fetch transliteration directly from Google API
        const response = await fetch(`https://inputtools.google.com/request?text=${encodeURIComponent(lastWord)}&itc=te-t-i0-und&num=1`);
        const data = await response.json();
        
        // Ensure successful response and valid data format
        if (data && data[0] === 'SUCCESS' && data[1] && data[1][0] && data[1][0][1]) {
          const teluguWord = data[1][0][1][0]; // Extract highest confidence Telugu word
          
          // Replace the English word with the Telugu word
          words[lastWordIdx] = teluguWord;
          
          // Re-assemble the text including the trigger character
          setTransliteratedText(words.join('') + lastChar);
        }
      } catch (error) {
        console.error("Transliteration fetch error:", error);
      }
    }
  }
}

export async function flushTransliteration(text: string): Promise<string> {
  if (!text) return text;

  // Split the text by non-alphanumeric to find the last true word token
  // Similar logic but doesn't require a trigger character
  const words = text.split(/([\s\n.,!?;:"')\]}]+)/);
  const lastWordIdx = words.length - 1;
  const lastWord = words[lastWordIdx];
  
  // Process if it's strictly an English alphabet word
  if (lastWord && /^[a-zA-Z]+$/.test(lastWord)) {
    try {
      // Fetch transliteration directly from Google API
      const response = await fetch(`https://inputtools.google.com/request?text=${encodeURIComponent(lastWord)}&itc=te-t-i0-und&num=1`);
      const data = await response.json();
      
      // Ensure successful response and valid data format
      if (data && data[0] === 'SUCCESS' && data[1] && data[1][0] && data[1][0][1]) {
        const teluguWord = data[1][0][1][0]; // Extract highest confidence Telugu word
        
        // Replace the English word with the Telugu word
        words[lastWordIdx] = teluguWord;
        
        // Re-assemble the text
        return words.join('');
      }
    } catch (error) {
      console.error("Transliteration flush fetch error:", error);
    }
  }
  
  return text;
}
