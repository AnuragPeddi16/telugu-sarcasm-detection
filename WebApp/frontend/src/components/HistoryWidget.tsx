import React, { useState } from 'react';
import styles from './HistoryWidget.module.css';

export interface HistoryItem {
    id: string;
    originalContext: string;
    sarcasticResponse?: string;
    nonSarcasticResponse?: string;
    isSarcastic?: boolean; // For validation
    sarcasticExample?: string; // For validation
    type: 'conversation' | 'validation';
}

interface HistoryWidgetProps {
    history: HistoryItem[];
    onEdit: (item: HistoryItem, index: number) => void;
    isOpen: boolean;
    onClose: () => void;
    theme: 'blue' | 'orange';
}

const HistoryWidget: React.FC<HistoryWidgetProps> = ({ history, onEdit, isOpen, onClose, theme }) => {
    if (!isOpen) return null;

    return (
        <div className={styles.overlay} onClick={onClose}>
            <div className={`${styles.drawer} ${theme === 'orange' ? styles.orangeTheme : styles.blueTheme}`} onClick={e => e.stopPropagation()}>
                <div className={styles.header}>
                    <h3>
                        History
                        <span className={styles.subtitle}>చరిత్ర</span>
                    </h3>
                    <button onClick={onClose} className={styles.closeButton}>&times;</button>
                </div>

                <div className={styles.list}>
                    {history.length === 0 ? (
                        <p className={styles.empty}>
                            No history yet. Start answering!
                            <br />
                            <span className={styles.teluguText}>ఇంకా చరిత్ర లేదు. సమాధానం ఇవ్వడం ప్రారంభించండి!</span>
                        </p>
                    ) : (
                        history.map((item, index) => (
                            <div key={item.id} className={styles.card}>
                                <div className={styles.cardHeader}>
                                    <span className={styles.index}>#{index + 1}</span>
                                    <button 
                                        className={styles.editButton}
                                        onClick={() => {
                                            onEdit(item, index);
                                            onClose();
                                        }}
                                    >
                                        Edit / సవరించు
                                    </button>
                                </div>
                                
                                <p className={styles.context}>
                                    <strong>{item.type === 'validation' ? 'Utterance:' : 'Context:'}</strong> {item.originalContext.replace(/\\n/g, ' ').substring(0, 50)}...
                                </p>

                                {item.type === 'conversation' && (
                                    <>
                                        <div className={styles.responseRow}>
                                            <span className={styles.label}>Sarcastic:</span>
                                            <span className={styles.value}>{item.sarcasticResponse || '(Audio)'}</span>
                                        </div>
                                        <div className={styles.responseRow}>
                                            <span className={styles.label}>Non-Sarcastic:</span>
                                            <span className={styles.value}>{item.nonSarcasticResponse || '(Audio)'}</span>
                                        </div>
                                    </>
                                )}

                                {item.type === 'validation' && (
                                    <>
                                        <div className={styles.responseRow}>
                                            <span className={styles.label}>Verdict:</span>
                                            <span className={`${styles.badge} ${item.isSarcastic ? styles.sarcastic : styles.nonSarcastic}`}>
                                                {item.isSarcastic ? 'Sarcastic' : 'Non-Sarcastic'}
                                            </span>
                                        </div>
                                        {!item.isSarcastic && item.sarcasticExample && (
                                            <div className={styles.responseRow}>
                                                <span className={styles.label}>Your Example:</span>
                                                <span className={styles.value}>{item.sarcasticExample}</span>
                                            </div>
                                        )}
                                    </>
                                )}
                            </div>
                        ))
                    )}
                </div>
            </div>
        </div>
    );
};

export default HistoryWidget;
