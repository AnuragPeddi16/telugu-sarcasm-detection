import React from 'react';
import styles from './Tutorial.module.css';

interface TutorialProps {
    onStart: () => void;
    type: 'conversation' | 'validation';
}

const Tutorial: React.FC<TutorialProps> = ({ onStart, type }) => {
    return (
        <div className={styles.overlay}>
            <div className={`${styles.modal} ${type === 'conversation' ? styles.blueTheme : styles.orangeTheme}`}>
                <h2 className={styles.title}>
                    {type === 'conversation' ? 'How to Participate' : 'How to Validate'}
                    <span className={styles.subtitle}>ఎలా పాల్గొనాలి / ఎలా ధృవీకరించాలి</span>
                </h2>
                
                <div className={styles.content}>
                    <div className={styles.section}>
                        <h3>
                            Example Context
                            <span className={styles.teluguText}>సందర్భం ఉదాహరణ</span>
                        </h3>
                        <p>
                            {type === 'conversation' ? (
                                <>
                                    "It started raining heavily just as we planned to go out."
                                    <br />
                                    <span className={styles.teluguText}>బయటికి వెళ్దాం అనుకున్నప్పుడే జోరుగా వర్షం మొదలైంది.</span>
                                </>
                            ) : (
                                <>
                                    <strong>Speaker A:</strong> "I got distracted by the phone and burnt the curry completely."
                                    <br />
                                    <span className={styles.teluguText}>ఫోన్ చూసుకుంటూ కూర మొత్తం మాడగొట్టేశాను.</span>
                                </>
                            )}
                        </p>
                    </div>

                    <div className={styles.examplesGrid}>
                        <div className={styles.exampleCard}>
                            <h4>
                                {type === 'conversation' ? 'Sarcastic Response' : 'Example 1: Sarcastic'}
                                <span className={styles.teluguText}>
                                    {type === 'conversation' ? 'వ్యంగ్య స్పందన' : 'ఉదాహరణ 1: వ్యంగ్యమైనది'}
                                </span>
                            </h4>
                            <p>
                                {type === 'conversation' ? (
                                    <>
                                        "Great! The weather gods must really love our plans."
                                        <br />
                                        <span className={styles.teluguText}>అద్భుతం! వరుణ దేవుడికి మన ప్లాన్ అంటే చాలా ఇష్టం అనుకుంటా.</span>
                                    </>
                                ) : (
                                    <>
                                        <strong>Speaker B:</strong> "Wonderful! So now we have the fortune of eating charcoal."
                                        <br />
                                        <span className={styles.teluguText}>అద్భుతం! ఇప్పుడు మనకి బొగ్గులు తినే అదృష్టం కలిగింది అన్నమాట.</span>
                                    </>
                                )}
                            </p>
                            <div className={styles.tag}>Sarcastic / వ్యంగ్యమైనది</div>
                            {type === 'validation' && (
                                <p className={styles.explanation}>
                                    <strong>Why:</strong> Using positive words like "Wonderful" for a negative event (burnt food).
                                    <br />
                                    <span className={styles.teluguText}>ఎందుకు: మాడిన కూర లాంటి చెడు సంఘటనకు "అద్భుతం" అనే పదం వాడటం.</span>
                                </p>
                            )}
                        </div>

                        <div className={styles.exampleCard}>
                            <h4>
                                {type === 'conversation' ? 'Non-Sarcastic Response' : 'Example 2: Non-Sarcastic'}
                                <span className={styles.teluguText}>
                                    {type === 'conversation' ? 'సాధారణ స్పందన' : 'ఉదాహరణ 2: సాధారణమైనది'}
                                </span>
                            </h4>
                            <p>
                                {type === 'conversation' ? (
                                    <>
                                        "Oh no, we should wait for a while until it stops."
                                        <br />
                                        <span className={styles.teluguText}>అయ్యో, వర్షం తగ్గే వరకు మనం కాసేపు ఆగాలి.</span>
                                    </>
                                ) : (
                                    <>
                                        <strong>Speaker B:</strong> "It's okay, let's order something from outside."
                                        <br />
                                        <span className={styles.teluguText}>పర్లేదులే, బయట నుంచి ఏదైనా ఆర్డర్ చేద్దాం.</span>
                                    </>
                                )}
                            </p>
                            <div className={styles.tag}>Non-Sarcastic / సాధారణమైనది</div>
                            {type === 'validation' && (
                                <p className={styles.explanation}>
                                    <strong>Why:</strong> A direct and practical solution without mockery.
                                    <br />
                                    <span className={styles.teluguText}>ఎందుకు: ఎగతాళి లేకుండా సూటిగా మరియు ఆచరణాత్మక పరిష్కారం చెప్పడం.</span>
                                </p>
                            )}
                        </div>
                    </div>

                    <div className={styles.instructions}>
                        <p>
                            {type === 'conversation' 
                                ? "Please provide both sarcastic and non-sarcastic responses for the given conversation."
                                : "Please decide if the response is sarcastic or not."}
                            <br />
                            <span className={styles.teluguText}>
                                {type === 'conversation'
                                    ? "దయచేసి ఇచ్చిన సంభాషణకు వ్యంగ్య మరియు సాధారణ సమాధానాలు రెండూ ఇవ్వండి."
                                    : "దయచేసి స్పందన వ్యంగ్యంగా ఉందో లేదో నిర్ణయించండి."}
                            </span>
                        </p>
                    </div>
                </div>

                <button className={styles.startButton} onClick={onStart}>
                    Start / ప్రారంభించు
                </button>
            </div>
        </div>
    );
};

export default Tutorial;
